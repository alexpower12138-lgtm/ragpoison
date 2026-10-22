"""The corpus audit (WP9): Stage-1 screening plus budgeted batched judging.

The audit answers a different question from the query-time detector. There the pool is
a handful of retrieved passages and the cost is bounded by the number of queries; here
the unit is the whole corpus and the cost is bounded by a token budget per scan. So the
design is:

1. **screen for free** with the five-component coarse score;
2. **judge in batches** of ten truncated passages, one request each;
3. **stop the moment the budget is spent**, and record the coverage that was reached.

Five strategies decide *which* chunks a scan looks at: everything, only what has not been
judged yet (incremental), only what the screen selected, one hash shard of the corpus, or
a combination of screen and shard.

The audit reports recall against ground truth, which is only possible because the
corpus is synthetic: a chunk whose id starts with ``p_atk`` is poisoned by construction.
That is stated wherever the number is reported, because on a real corpus this audit
could not know.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from ..config import Config, load_config
from ..datagen.loader import load_bundle
from ..llm import mock as mock_responders
from ..llm import prompts
from ..llm.client import Call, LLMClient
from ..llm.parse import parse_flags
from ..paths import path as repo_path, rel
from . import coarse

STRATEGIES: tuple[str, ...] = (
    "full",
    "incremental",
    "coarse_to_fine",
    "sharded",
    "coarse_to_fine+sharded",
)

SCAN_VERSION = "1.0.0"

#: How many requests to submit before checking the token budget again. Small enough
#: that a scan cannot overshoot its budget by much, large enough to keep the run quick.
WINDOW = 20


def _shard_of(chunk_id: str, shards: int) -> int:
    digest = hashlib.sha1(chunk_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % max(1, shards)


def _truncate(text: str, words: int) -> str:
    parts = text.split()
    return " ".join(parts[:words]) + (" ..." if len(parts) > words else "")


def load_policy(name: str) -> dict:
    path = repo_path("configs", f"scan_{name}.yaml")
    if not path.exists():
        raise FileNotFoundError(f"no such scan policy: {path}")
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload.get("scan", {})


def _status_path(seed: int, policy: str = "") -> Path:
    """One status file per policy.

    Sharing one file across the five policies would make their coverage and recall
    cumulative - the second policy would inherit the first one's verdicts and look far
    better than it is - so each policy keeps its own state. The incremental policy is
    the exception by design: it is *meant* to read what earlier rounds of itself left.
    """
    suffix = f"_{policy}" if policy else ""
    return repo_path("results", "scans", f"chunk_status_seed{seed}{suffix}.json")


def load_status(seed: int, policy: str = "") -> dict:
    path = _status_path(seed, policy)
    if not path.exists():
        return {"scan_version": SCAN_VERSION, "chunks": {}, "dirty": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("scan_version") != SCAN_VERSION:
        # A version change invalidates every stored verdict.
        return {"scan_version": SCAN_VERSION, "chunks": {}, "dirty": list(payload.get("chunks", {}))}
    return payload


def save_status(seed: int, status: dict, policy: str = "") -> str:
    path = _status_path(seed, policy)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(status, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return rel(path)


def select_chunks(
    *,
    strategy: str,
    ids: list[str],
    scores: dict[str, dict],
    status: dict,
    tau_c: float,
    shards: int,
    shard: int,
) -> list[str]:
    """Which chunks this scan should look at, in a deterministic order."""
    judged = status.get("chunks", {})
    dirty = set(status.get("dirty", []))
    selected = []
    for chunk_id in ids:
        if strategy in ("coarse_to_fine", "coarse_to_fine+sharded"):
            if scores.get(chunk_id, {}).get("coarse", 0.0) <= tau_c:
                continue
        if strategy in ("sharded", "coarse_to_fine+sharded"):
            if _shard_of(chunk_id, shards) != shard:
                continue
        if strategy == "incremental":
            if chunk_id in judged and chunk_id not in dirty:
                continue
        selected.append(chunk_id)
    return selected


def rebuild_index(seed: int | None = None) -> str:
    """One index entry per policy: the run that covered the most.

    Every invocation appends, so a budget-limited trial followed by a full run leaves
    two entries for the same policy and the dashboard shows both. Keeping the largest
    run per policy is what makes the table readable, and it is still derived entirely
    from the per-scan files, so nothing is lost.
    """
    directory = repo_path("results", "scans")
    directory.mkdir(parents=True, exist_ok=True)
    best: dict[tuple, dict] = {}
    for path in sorted(directory.glob("scan_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if payload.get("mock"):
            continue
        if seed is not None and int(payload.get("seed", -1)) != seed:
            continue
        key = (payload.get("policy"), int(payload.get("seed", 0)))
        current = best.get(key)

        def rank(entry: dict) -> tuple:
            # Prefer more coverage, and on a tie prefer the file that carries the newer
            # within-scanned recall field, which older runs predate.
            return (
                int(entry.get("chunks_judged", 0)),
                1 if "recall_within_scanned" in entry else 0,
            )

        if current is None or rank(payload) > rank(current):
            best[key] = payload

    entries = []
    for (policy, seed_value), payload in sorted(best.items()):
        entries.append(
            {
                "scan_id": payload["scan_id"],
                "policy": policy,
                "seed": seed_value,
                "strategy": payload.get("strategy"),
                "covered": payload.get("chunks_judged", 0),
                "coverage": payload.get("coverage", 0.0),
                "tokens_used": payload.get("tokens_used", 0),
                "suspicious_count": payload.get("suspicious_count", 0),
                "recall": payload.get("recall_against_ground_truth", 0.0),
                "recall_within_scanned": payload.get("recall_within_scanned", 0.0),
                "precision": payload.get("precision_against_ground_truth", 0.0),
                "tokens_per_chunk": payload.get("tokens_per_chunk", 0.0),
                "budget_tokens": payload.get("budget_tokens"),
            }
        )
    index_path = directory / "index.json"
    index_path.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return rel(index_path)


def run_scan(
    seed: int = 42,
    policy_name: str = "P1",
    *,
    rounds: int = 1,
    config: Config | None = None,
    budget_tokens: int | None = None,
    tau_c: float | None = None,
    strategy: str | None = None,
    shard: int = 0,
    mock: bool = False,
    write: bool = True,
    quiet: bool = False,
) -> dict:
    cfg = config or load_config("default.yaml")
    policy = load_policy(policy_name)
    batches_of = int(policy.get("batch_size") or prompts.SCAN_BATCH_SIZE)
    strategy = strategy or str(policy.get("strategy") or "full")
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy}")
    tau = float(tau_c if tau_c is not None else policy.get("coarse_threshold") or 0.4)
    shards = int(policy.get("shard_count") or 1)
    budget = int(budget_tokens if budget_tokens is not None else policy.get("budget_tokens_per_scan") or 50000)

    bundle = load_bundle(seed)
    texts = bundle.texts()
    ids = sorted(texts)
    calibration_path = repo_path("cache", f"calibration_seed{seed}.json")
    ppl_statistics = None
    if calibration_path.exists():
        ppl_statistics = json.loads(calibration_path.read_text(encoding="utf-8")).get("s6")
    scores = coarse.coarse_scores(texts, ppl_statistics=ppl_statistics)

    status = load_status(seed, policy_name)
    client = LLMClient(
        cfg,
        mock=mock,
        mock_responder=mock_responders.signal_responder,
        quiet=quiet,
    )

    rounds_done = []
    tokens_used = 0
    unparsed_chunks: set[str] = set()
    for round_index in range(rounds):
        chosen = select_chunks(
            strategy=strategy,
            ids=ids,
            scores=scores,
            status=status,
            tau_c=tau,
            shards=shards,
            shard=(shard + round_index) % max(1, shards),
        )
        if not chosen:
            rounds_done.append({"round": round_index + 1, "chunks": 0, "tokens": 0, "note": "nothing left to scan"})
            break

        batches = [chosen[index : index + batches_of] for index in range(0, len(chosen), batches_of)]
        batch_tokens = 0
        batch_calls = 0
        batch_flagged: Counter = Counter()
        batch_judged = 0
        stopped = False

        estimated = 0.0
        # Advance by the number of batches actually submitted. Stepping the loop by
        # WINDOW while the window could be smaller silently skipped the remainder: with
        # thirteen batches and a first window of one, twelve were never sent.
        cursor = 0
        while cursor < len(batches):
            spent = tokens_used + batch_tokens
            if spent >= budget:
                stopped = True
                break
            if estimated > 0 and budget - spent < estimated:
                stopped = True
                break
            allowance = 1 if estimated <= 0 else max(1, int((budget - spent) / estimated))
            window = batches[cursor : cursor + min(WINDOW, allowance)]
            cursor += len(window)
            calls = []
            for chunk_batch in window:
                passages = "\n".join(
                    f"[{position}] {_truncate(texts[chunk_id], prompts.SCAN_PASSAGE_WORDS)}"
                    for position, chunk_id in enumerate(chunk_batch, start=1)
                )
                calls.append(
                    Call(
                        system=prompts.SCAN_JUDGE_SYSTEM,
                        user=prompts.SCAN_JUDGE_USER.format(passages=passages),
                        stage=f"scan_{policy_name}",
                        max_tokens=int(cfg.get("llm.max_tokens_json")) * 3,
                        extra={"seed": seed, "policy": policy_name},
                    )
                )
            results = client.complete_many(calls, progress_every=max(len(calls), 1))
            window_tokens = sum(
                (result.prompt_tokens or 0) + (result.completion_tokens or 0) for result in results
            )
            if results:
                estimated = (window_tokens / len(results)) if window_tokens else estimated
            for chunk_batch, result in zip(window, results):
                batch_tokens += (result.prompt_tokens or 0) + (result.completion_tokens or 0)
                batch_calls += 1
                expected = list(range(1, len(chunk_batch) + 1))
                parsed = parse_flags(result.text, expected) if result.ok else None
                if parsed is None:
                    # A failed batch is NOT recorded as genuine: an unreadable verdict is
                    # not evidence of innocence. The chunks go on the dirty list so an
                    # incremental scan retries them, and coverage stays honest.
                    status["dirty"] = sorted(set(status.get("dirty", [])) | set(chunk_batch))
                    unparsed_chunks.update(chunk_batch)
                    continue
                for position, chunk_id in enumerate(chunk_batch, start=1):
                    harmful = int(parsed.get(position, 0))
                    status.setdefault("chunks", {})[chunk_id] = "harmful" if harmful else "genuine"
                    batch_flagged[chunk_id] = harmful
                    batch_judged += 1

        tokens_used += batch_tokens
        rounds_done.append(
            {
                "round": round_index + 1,
                "shard": (shard + round_index) % max(1, shards),
                "chunks": batch_judged,
                "calls": batch_calls,
                "tokens": batch_tokens,
                "stopped_by_budget": stopped,
            }
        )
        if stopped:
            break

    judged_chunks = sorted(status.get("chunks", {}))
    harmful = {chunk for chunk, verdict in status.get("chunks", {}).items() if verdict == "harmful"}
    poison = {chunk for chunk in ids if chunk.startswith("p_atk")}
    coverage = len(judged_chunks) / len(ids) if ids else 0.0
    recall = len(harmful & poison) / len(poison) if poison else 0.0
    precision = len(harmful & poison) / len(harmful) if harmful else 0.0
    # Two recalls, because one number hides the difference between "the detector is
    # weak" and "the scan only looked at part of the corpus". The corpus recall is what
    # the audit ultimately delivers; the within-scanned recall is the detector's quality
    # on what it did look at.
    scanned_poison = poison & set(judged_chunks)
    recall_scanned = len(harmful & scanned_poison) / len(scanned_poison) if scanned_poison else 0.0

    payload = {
        "scan_id": f"{policy_name}-seed{seed}-{int(time.time())}",
        "policy": policy_name,
        "seed": seed,
        "strategy": strategy,
        "shards": shards,
        "tau_c": tau,
        "batch_size": batches_of,
        "budget_tokens": budget,
        "scan_version": SCAN_VERSION,
        "coarse": {
            "above_tau": sum(1 for value in scores.values() if value["coarse"] > tau),
            "total": len(ids),
            "precision": (
                sum(1 for chunk in ids if scores[chunk]["coarse"] > tau and chunk in poison)
                / max(1, sum(1 for chunk in ids if scores[chunk]["coarse"] > tau))
            ),
            "recall": (
                sum(1 for chunk in ids if scores[chunk]["coarse"] > tau and chunk in poison)
                / max(1, len(poison))
            ),
        },
        "rounds": rounds_done,
        "tokens_used": tokens_used,
        "chunks_judged": len(judged_chunks),
        "coverage": round(coverage, 6),
        "suspicious_count": len(harmful),
        "recall_against_ground_truth": round(recall, 6),
        "recall_within_scanned": round(recall_scanned, 6),
        "poison_in_scanned": len(scanned_poison),
        "precision_against_ground_truth": round(precision, 6),
        "tokens_per_chunk": round(tokens_used / len(judged_chunks), 2) if judged_chunks else 0.0,
        "unparsed_chunks": sorted(unparsed_chunks),
        "mock": mock,
        "llm": client.summary(),
    }

    if write:
        out = repo_path("results", "scans")
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"scan_{payload['scan_id']}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        payload["path"] = rel(path)
        payload["status_path"] = save_status(seed, status, policy_name)
        payload["index"] = rebuild_index(seed)

    return payload
