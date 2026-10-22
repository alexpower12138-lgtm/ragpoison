"""Data access for the dashboard. Pure Python: no streamlit, no matplotlib.

Everything the five tabs display is produced here, from files that already exist.
Two consequences are deliberate:

* every function returns an empty structure rather than raising when its file is
  missing, so the dashboard renders an explanation instead of a traceback on a fresh
  checkout;
* the tab layout can be tested without a browser, which is the only way the visual
  layer gets any verification at all in this project.

The prompt inspector does not replay stored prompts, because the ledger never stored
them. Instead it *rebuilds* the prompt from the deterministic pipeline and looks the
result up in the answer cache by hash. That is a stronger claim than showing a stored
copy: it demonstrates that the prompt the code produces today is the one the run paid
for.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import load_config
from ..paths import path as repo_path
from ..signals.load import load_joined, row_key

SIGNAL_LABELS: dict[str, str] = {
    "query": "S1 similarity over-fit",
    "cross": "S2 cross-passage consistency",
    "conflict": "S3 answer centrality",
    "conflict_majority": "S3b majority vote",
    "instruction": "S4 instruction traces",
    "multiview": "S5 four-view judgement",
    "ppl": "S6 pseudo-perplexity",
}

SHORT: dict[str, str] = {
    "query": "S1",
    "cross": "S2",
    "conflict": "S3",
    "conflict_majority": "S3b",
    "instruction": "S4",
    "multiview": "S5",
    "ppl": "S6",
}


# --------------------------------------------------------------------------- #
# Small file helpers
# --------------------------------------------------------------------------- #


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    import csv

    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _tail_jsonl(path: Path, limit: int) -> list[dict]:
    """Read the last ``limit`` rows without loading a 25 MB file into memory."""
    if not path.exists():
        return []
    from collections import deque

    with path.open(encoding="utf-8") as handle:
        window = deque(handle, maxlen=limit)
    rows = []
    for line in window:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


# --------------------------------------------------------------------------- #
# Availability
# --------------------------------------------------------------------------- #


def available_seeds() -> list[int]:
    seeds = set()
    for name in ("run_full", "run_replicate", "run_pilot"):
        for path in repo_path("results").glob(f"{name}_seed*.json"):
            try:
                seeds.add(int(path.stem.split("seed")[-1]))
            except ValueError:
                continue
    return sorted(seeds)


def available_runs(seed: int) -> list[str]:
    """Which run modes have a report for this seed."""
    modes = []
    for mode in ("pilot", "full", "replicate"):
        if repo_path("results", f"run_{mode}_seed{seed}.json").exists():
            modes.append(mode)
    return modes


def run_report(seed: int, mode: str | None = None) -> dict:
    candidates = [mode] if mode else ["replicate", "full", "pilot"]
    for name in candidates:
        payload = _read_json(repo_path("results", f"run_{name}_seed{seed}.json"))
        if payload:
            payload["_mode"] = name
            return payload
    return {}


# --------------------------------------------------------------------------- #
# Tab 1 - prompt inspector
# --------------------------------------------------------------------------- #


def ledger(limit: int = 400) -> list[dict]:
    return _tail_jsonl(repo_path("logs", "llm_calls.jsonl"), limit)


def ledger_summary(rows: list[dict]) -> dict:
    total = len(rows)
    paid = sum(1 for row in rows if not row.get("cache_hit"))
    tokens_in = sum(int(row.get("prompt_tokens") or 0) for row in rows)
    tokens_out = sum(int(row.get("completion_tokens") or 0) for row in rows)
    by_stage: dict[str, int] = {}
    for row in rows:
        stage = str(row.get("stage") or "?")
        by_stage[stage] = by_stage.get(stage, 0) + 1
    errors = sum(1 for row in rows if row.get("ok") is False)
    return {
        "rows": total,
        "paid": paid,
        "cache_hits": total - paid,
        "cache_rate": (total - paid) / total if total else 0.0,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "by_stage": dict(sorted(by_stage.items(), key=lambda item: -item[1])),
        "errors": errors,
    }


def rebuild_prompt(
    seed: int,
    stage: str,
    *,
    corpus_id: str | None = None,
    doc_id: str | None = None,
    condition: str = "plain",
) -> dict:
    """Reconstruct one request from the pipeline and look it up in the cache.

    Supported stages are the three signal prompts and the baseline generation prompt,
    which between them cover every paid call the project makes.
    """
    from ..datagen.loader import load_bundle
    from ..llm import prompts
    from ..llm.client import Call, LLMClient
    from ..signals.extract import build_context
    from ..datagen.corpora import iter_query_corpora

    cfg = load_config("default.yaml")
    bundle = load_bundle(seed)
    texts = bundle.texts()
    queries = {row["query_id"]: row for row in bundle.queries}

    target = None
    for corpus in iter_query_corpora(bundle):
        if corpus_id and corpus["corpus_id"] == corpus_id:
            target = corpus
            break
    if target is None:
        return {"error": f"corpus {corpus_id} not found in seed {seed}"}

    query = queries[target["query_id"]]
    context, _bm25, _tfidf = build_context(bundle, target, query, {}, cfg, int(cfg.get("retrieval.top_n")))
    max_tokens_json = int(cfg.get("llm.max_tokens_json"))

    call = None
    if stage == "audit_baseline_plain" or stage == "audit_baseline_guarded":
        system = prompts.GEN_PLAIN_SYSTEM if stage.endswith("plain") else prompts.GEN_GUARDED_SYSTEM
        call = Call(
            system=system,
            user=prompts.GEN_USER.format(
                question=query["question"],
                context=prompts.format_context([texts[c] for c in context.ids()[: int(cfg.get("retrieval.top_k"))]]),
            ),
            stage=stage,
            max_tokens=int(cfg.get("llm.max_tokens_answer")),
        )
    else:
        candidate = None
        for item in context.candidates:
            if item.doc_id == doc_id:
                candidate = item
                break
        if candidate is None:
            return {"error": f"document {doc_id} is not in the retrieved pool of {corpus_id}"}
        if stage == "s3_answer":
            call = Call(
                system=prompts.S3_SYSTEM,
                user=prompts.S3_USER.format(passage=candidate.text, question=query["question"]),
                stage=stage,
                max_tokens=max_tokens_json,
            )
        elif stage == "s4_instruction":
            call = Call(
                system=prompts.S4_SYSTEM,
                user=prompts.S4_USER.format(passage=candidate.text),
                stage=stage,
                max_tokens=max_tokens_json,
            )
        elif stage.startswith("s5_"):
            view = stage[3:]
            # Ranks are 1-based, so the batch index is (rank - 1) // 5 on both sides;
            # using rank // 5 on one side splits the batch at the wrong boundary and the
            # rebuilt prompt then hashes differently from the one that was paid for.
            batch_index = (candidate.rank - 1) // 5
            batch = [c for c in context.candidates if (c.rank - 1) // 5 == batch_index]
            passages = "\n".join(
                f"[{position}] {item.text}" for position, item in enumerate(batch, start=1)
            )
            call = Call(
                system=prompts.S5_SYSTEM,
                user=prompts.S5_USER.format(
                    question=query["question"],
                    passages=passages,
                    view_description=prompts.S5_VIEWS.get(view, ""),
                ),
                stage=stage,
                max_tokens=max_tokens_json * 5,
            )

    if call is None:
        return {"error": f"stage {stage} is not reconstructable"}

    client = LLMClient(cfg, quiet=True)
    key = client._cache_key(call)
    path = client._cache_path(key)
    cached = _read_json(path)
    return {
        "stage": stage,
        "system": call.system,
        "user": call.user,
        "max_tokens": call.max_tokens,
        "cache_key": key,
        "cache_hit": bool(cached),
        "response": cached.get("text", ""),
        "prompt_tokens": cached.get("prompt_tokens"),
        "completion_tokens": cached.get("completion_tokens"),
    }


# --------------------------------------------------------------------------- #
# Tab 2 - detection dashboard
# --------------------------------------------------------------------------- #


def ablation() -> list[dict]:
    rows = _read_csv(repo_path("results", "summary_ablation.csv"))
    for row in rows:
        for key, value in list(row.items()):
            if key in ("config", "label", "signals", "rule"):
                continue
            try:
                row[key] = float(value) if value not in ("", "None") else None
            except (TypeError, ValueError):
                pass
        if isinstance(row.get("seed"), str):
            try:
                row["seed"] = int(row["seed"])
            except ValueError:
                pass
    return rows


def ablation_details() -> dict[tuple, dict]:
    payload = _read_json(repo_path("results", "summary_ablation_details.json"))
    if not isinstance(payload, list):
        return {}
    return {(int(item.get("seed", 0)), item.get("config")): item for item in payload}


def defence(seed: int) -> dict:
    payload = _read_json(repo_path("results", f"defence_seed{seed}.json"))
    return payload.get("configs", {}) if payload else {}


def defence_cells(seed: int) -> list[dict]:
    return _tail_jsonl(repo_path("results", f"defence_seed{seed}_cells.jsonl"), 20000)


# --------------------------------------------------------------------------- #
# Tab 3 - signal x attack
# --------------------------------------------------------------------------- #


def signal_attack(seed: int) -> dict:
    """Per-family AUC and per-label mean for every signal, from the run report."""
    report = run_report(seed)
    if not report:
        return {}
    return {
        "per_family_auc": report.get("per_family_auc", {}),
        "per_label": report.get("per_label", {}),
        "auc": report.get("auc", {}),
        "auc_hard": report.get("auc_hard", {}),
        "n_hard": report.get("n_hard"),
        "n_rows": report.get("n_rows"),
    }


# --------------------------------------------------------------------------- #
# Tab 4 - per-query inspector
# --------------------------------------------------------------------------- #


def query_index(seed: int) -> list[str]:
    rows = load_joined(seed) if repo_path("results", "signals", f"seed{seed}").exists() else []
    seen = []
    for row in rows:
        key = f"{row['attack_family']}_inj{row['injection']}_{row['query_id']}"
        if key not in seen:
            seen.append(key)
    return seen


def query_detail(seed: int, label: str) -> dict:
    """One pool with passage text, every signal, and the stored answer."""
    from ..datagen.loader import load_bundle

    rows = load_joined(seed)
    if not rows:
        return {}
    selected = [
        row
        for row in rows
        if f"{row['attack_family']}_inj{row['injection']}_{row['query_id']}" == label
    ]
    if not selected:
        return {}
    bundle = load_bundle(seed)
    texts = bundle.texts()
    queries = {row["query_id"]: row for row in bundle.queries}
    query = queries[selected[0]["query_id"]]
    selected.sort(key=lambda row: row["rank"])
    return {
        "query_id": query["query_id"],
        "question": query["question"],
        "correct_answer": query.get("correct_answer"),
        "target_answer": query.get("target_answer"),
        "split": query.get("split"),
        "corpus_id": selected[0]["corpus_id"],
        "attack_family": selected[0]["attack_family"],
        "injection": selected[0]["injection"],
        "rows": [
            {
                "rank": row["rank"],
                "doc_id": row["doc_id"],
                "label": row["label_class"],
                "is_attack": bool(row["is_attack_for_query"]),
                "text": texts.get(row["doc_id"], ""),
                "answer": row.get("answer", ""),
                **{
                    name: float(row.get(column, 0.0))
                    for name, column in (
                        ("s1", "s1"),
                        ("s2", "s2"),
                        ("s3", "s3"),
                        ("s3b", "s3b"),
                        ("s4", "s4"),
                        ("s5", "s5"),
                        ("s6", "s6"),
                        ("s7", "s7"),
                    )
                },
            }
            for row in selected
        ],
    }


def fuse_rows(rows: list[dict], weights: dict[str, float], threshold: float) -> list[dict]:
    """Score the visible pool with the dashboard's own weights. Free, instant."""
    total = sum(abs(value) for value in weights.values()) or 1.0
    scored = []
    for row in rows:
        score = 0.0
        for name, weight in weights.items():
            score += weight * float(row.get(name, 0.0))
        scored.append({**row, "score": score / total, "flagged": (score / total) > threshold})
    return scored


# --------------------------------------------------------------------------- #
# Tab 5 - security scan
# --------------------------------------------------------------------------- #


def scan_strategies() -> list[dict]:
    import yaml

    out = []
    for path in sorted(repo_path("configs").glob("scan_P*.yaml")):
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - the dashboard must not die on config
            out.append({"name": path.stem, "error": str(exc)})
            continue
        scan = payload.get("scan", {}) if isinstance(payload, dict) else {}
        out.append({"name": path.stem, **scan})
    return out


def scans() -> list[dict]:
    index = repo_path("results", "scans", "index.json")
    payload = _read_json(index)
    if isinstance(payload, list):
        return payload
    return []


#: Relative API cost of one instance, in calls, per signal. S3 asks once per
#: candidate (15), S4 asks once per distinct passage (about 1 after dedupe across the
#: matrix), S5 asks once per batch of five per view (9).
SIGNAL_COST: dict[str, int] = {
    "query": 0,
    "cross": 0,
    "conflict": 15,
    "conflict_majority": 15,
    "instruction": 1,
    "multiview": 9,
    "ppl": 0,
}


def cost_units(signals: list[str]) -> int:
    """API calls per instance for a configuration, for the cost-accuracy view."""
    return sum(SIGNAL_COST.get(name, 0) for name in signals)


def sweep(seed: int, config_id: str, points: int = 40) -> dict:
    """F1 and the precision-recall curve against the threshold, for one configuration.

    Computed here rather than stored, because it is a property of one configuration
    and one seed and costs a single sort over 12600 rows.
    """
    from ..config import load_ablation_matrix
    from ..fusion import calibrate as calibration
    from ..fusion.fuse import build_model

    matrix = load_ablation_matrix()
    spec = matrix.get(config_id)
    if spec is None:
        return {}
    rows = load_joined(seed)
    if not rows:
        return {}
    calib = [row for row in rows if row["split"] == "calib"]
    test = [row for row in rows if row["split"] == "test"]
    model = build_model(spec.signals, spec.fusion, flip_query_sign=spec.flip_query_sign)
    if spec.fusion == "weighted":
        calibration.fit_alphas(model, calib)
    elif spec.fusion == "logistic":
        calibration.fit_logistic(model, calib)
    threshold = calibration.fit_threshold(model, calib) if spec.fusion != "none" else None
    scored = [(model.score(row), int(row["is_attack_for_query"])) for row in test]
    scored.sort(reverse=True)
    positives = sum(label for _score, label in scored)
    taus: list[float] = []
    f1s: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    tp = fp = 0
    step = max(1, len(scored) // points)
    for index, (score, label) in enumerate(scored):
        if label:
            tp += 1
        else:
            fp += 1
        if index % step:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / positives if positives else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        taus.append(score)
        f1s.append(f1)
        precisions.append(precision)
        recalls.append(recall)
    return {
        "taus": taus,
        "f1": f1s,
        "precision": precisions,
        "recall": recalls,
        "operating": threshold.chosen if threshold else 0.0,
        "f1_at_operating": threshold.f1_score if threshold else 0.0,
    }


def corpus_size(seed: int) -> int:
    from ..datagen.loader import load_bundle

    try:
        return len(load_bundle(seed).texts())
    except Exception:  # noqa: BLE001
        return 0
