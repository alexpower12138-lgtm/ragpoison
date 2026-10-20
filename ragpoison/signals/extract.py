"""Frozen signal extraction: the expensive artefact everything else reads.

One row per (instance, candidate). An instance is one query evaluated against one
knowledge base, i.e. (seed, attack family, injection, query). The table is written
once and every ablation is computed from it offline, which is the property that
keeps the whole experiment affordable (decision D12 in the spec).

Every locally computable signal is evaluated for every row, even when the
configuration disables it: they cost nothing, and the ablation needs them.
``enabled`` records what the configuration asked for; the fusion step decides.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import Config, load_config
from ..datagen.corpora import iter_query_corpora
from ..datagen.loader import Bundle, load_bundle
from ..paths import path as repo_path
from ..retrieval.bm25 import BM25
from ..retrieval.tfidf import TfidfIndex
from .base import Candidate, SignalContext, run_signals

#: Signals this work package produces. The LLM-backed ones arrive in WP4.
LOCAL_SIGNALS: tuple[str, ...] = ("s1", "s2", "s6", "s7")

LABEL_CLASSES = ("poison_targeted", "cross_query_poison", "clean", "distractor")


def _label(doc_id: str, corpus: dict, bundle: Bundle) -> tuple[str, int]:
    """Label a candidate with the semantics fixed in decision D8."""
    owner = corpus["poison_owner"].get(doc_id)
    if owner is not None:
        return ("poison_targeted", 1) if owner == corpus["query_id"] else ("cross_query_poison", 0)
    if doc_id.startswith("p_clean_"):
        return "clean", 0
    return "distractor", 0


def _enabled(config: Config) -> dict[str, bool]:
    return {
        "s1": bool(config.get("detectors.enable_query_sim")),
        "s2": bool(config.get("detectors.enable_cross_consistency")),
        "s6": bool(config.get("detectors.enable_pseudo_ppl")),
        "s7": True,  # a fusion weight, never a verdict
    }


def build_context(
    bundle: Bundle,
    corpus: dict,
    query: dict,
    calibration: dict,
    config: Config,
    top_n: int,
) -> tuple[SignalContext, BM25, TfidfIndex]:
    texts = bundle.texts()
    documents = [(chunk_id, texts[chunk_id]) for chunk_id in corpus["chunk_ids"]]
    bm25 = BM25(
        k1=float(config.get("retrieval.bm25_k1")),
        b=float(config.get("retrieval.bm25_b")),
        lowercase=bool(config.get("retrieval.lowercase")),
        strip_punct=bool(config.get("retrieval.strip_punct")),
        drop_stopwords=bool(config.get("retrieval.drop_stopwords")),
    ).fit(documents)
    tfidf = TfidfIndex().fit(documents)

    ranked = bm25.ranked_ids(query["question"], top_n=top_n)
    candidates = [
        Candidate(doc_id=doc_id, text=texts[doc_id], rank=rank, bm25=score)
        for rank, (doc_id, score) in enumerate(ranked, start=1)
    ]
    context = SignalContext(
        config=config,
        query=query["question"],
        candidates=candidates,
        bm25=bm25,
        tfidf=tfidf,
        calibration=calibration,
        corpus_size=len(documents),
    )
    return context, bm25, tfidf


def extract_instance(
    bundle: Bundle,
    corpus: dict,
    query: dict,
    calibration: dict,
    config: Config,
    top_n: int,
) -> list[dict]:
    """Return the rows for one instance (one query against one corpus)."""
    context, _bm25, _tfidf = build_context(bundle, corpus, query, calibration, config, top_n)
    results = run_signals(context)
    enabled = _enabled(config)
    corpus_poison = set(corpus["poison_chunk_ids"])
    retrieved = [candidate.doc_id for candidate in context.candidates]
    atr = sum(1 for doc_id in retrieved if doc_id in corpus_poison) / len(retrieved)

    rows: list[dict] = []
    for candidate in context.candidates:
        label_class, is_attack = _label(candidate.doc_id, corpus, bundle)
        row = {
            "seed": bundle.seed,
            "attack_family": corpus["attack_family"],
            "injection": int(corpus["injection"]),
            "corpus_id": corpus["corpus_id"],
            "query_id": query["query_id"],
            "fact_id": query["fact_id"],
            "split": query["split"],
            "doc_id": candidate.doc_id,
            "rank": candidate.rank,
            "bm25": round(candidate.bm25, 6),
            "label_class": label_class,
            "is_attack_for_query": is_attack,
            "n_chunks": corpus["n_chunks"],
            "atr": round(atr, 6),
        }
        for name in LOCAL_SIGNALS:
            result = results[name]
            row[name] = round(result.value(candidate.doc_id), 6)
        row["enabled"] = ",".join(name for name in LOCAL_SIGNALS if enabled.get(name, False))
        rows.append(row)
    return rows


def extract(
    seed: int = 42,
    *,
    family: str | None = None,
    injection: int | None = None,
    query_id: str | None = None,
    query_ids: list[str] | None = None,
    limit: int | None = None,
    config: Config | None = None,
    top_n: int | None = None,
    write: bool = True,
    out_dir: Path | None = None,
) -> dict:
    """Extract the local signal table, optionally writing one file per corpus."""
    cfg = config or load_config("default.yaml")
    pool = int(top_n or cfg.get("retrieval.top_n"))
    bundle = load_bundle(seed)
    allowed = set(query_ids) if query_ids else None

    from .calibration import compute as compute_calibration

    calibration = compute_calibration(seed, top_n=pool)

    target_dir = out_dir or repo_path("results", "signals", f"seed{seed}")
    instances = 0
    written: list[str] = []
    rows_by_file: dict[str, list[dict]] = {}
    sample_rows: list[dict] = []

    for corpus in iter_query_corpora(bundle):
        if family and corpus["attack_family"] != family:
            continue
        if injection is not None and int(corpus["injection"]) != injection:
            continue
        if query_id and corpus["query_id"] != query_id:
            continue
        if allowed is not None and corpus["query_id"] not in allowed:
            continue
        if limit is not None and instances >= limit:
            break

        query = next(row for row in bundle.queries if row["query_id"] == corpus["query_id"])
        rows = extract_instance(bundle, corpus, query, calibration, cfg, pool)
        if not sample_rows:
            sample_rows = rows
        instances += 1
        key = f"{corpus['attack_family']}_inj{int(corpus['injection'])}"
        rows_by_file.setdefault(key, []).extend(rows)

    if write:
        target_dir.mkdir(parents=True, exist_ok=True)
        for key, rows in sorted(rows_by_file.items()):
            path = target_dir / f"{key}.jsonl"
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written.append(path.relative_to(repo_path()).as_posix())

    return {
        "seed": seed,
        "instances": instances,
        "rows": sum(len(rows) for rows in rows_by_file.values()),
        "calibration": calibration,
        "files": written,
        "sample_rows": sample_rows,
        "by_corpus": {key: len(rows) for key, rows in sorted(rows_by_file.items())},
    }


def separability(seed: int = 42, path: Path | None = None) -> list[dict]:
    """Mean signal value per label, a first look at which signal fires where."""
    target = path or repo_path("results", "signals", f"seed{seed}")
    buckets: dict[tuple[str, int, str], dict[str, list[float]]] = {}
    for candidate in sorted(target.glob("*.jsonl")):
        with candidate.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                key = (row["attack_family"], int(row["injection"]), row["label_class"])
                slot = buckets.setdefault(key, {name: [] for name in LOCAL_SIGNALS})
                for name in LOCAL_SIGNALS:
                    slot[name].append(float(row[name]))

    out: list[dict] = []
    for (family, injection, label_class), values in sorted(buckets.items()):
        out.append(
            {
                "attack_family": family,
                "injection": injection,
                "label_class": label_class,
                "n": len(values["s7"]),
                **{
                    f"mean_{name}": (sum(values[name]) / len(values[name]) if values[name] else 0.0)
                    for name in LOCAL_SIGNALS
                },
            }
        )
    return out
