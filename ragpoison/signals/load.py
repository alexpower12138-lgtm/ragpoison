"""Join the two signal tables into one row set.

The local table (S1/S2/S6/S7) and the LLM table (S3/S3b/S4/S5) are written by
different stages with different costs, so they are kept apart on disk and joined
here on ``(corpus_id, query_id, doc_id)``.

A row that exists in the local table but not in the LLM table gets zeros and is
marked ``miss``. That cannot normally happen in a completed run, and the caller is
expected to report the count rather than let it pass silently.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..paths import path as repo_path
from .base import ABLATION_COLUMNS

LLM_COLUMNS: tuple[str, ...] = ("s3", "s3b", "s4", "s5")
#: Extra per-row facts carried from the LLM table, not signals themselves.
LLM_EXTRAS: tuple[str, ...] = ("s3_c", "s3_unknown", "answer", "s4_category", "call_failed")


def row_key(row: dict) -> tuple:
    """The unique identity of one candidate.

    ``doc_id`` alone is **not** unique. The same base-pool passage is retrieved for
    many queries, and within one query the same passage appears once per attack
    family and once per injection size: the 12600 test candidates share only 1484
    distinct document ids, with one id appearing 198 times.

    A dictionary keyed by ``doc_id`` silently keeps one of those values. That is
    exactly the bug that made an earlier version of the ablation report the
    false-positive rate of one configuration while filtering with the scores of
    another, so every score lookup goes through this function.
    """
    return (row["corpus_id"], row["query_id"], row["doc_id"])


def _cells(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_joined(seed: int, directory: Path | None = None) -> list[dict]:
    """Return every candidate row of one seed with all signal columns present."""
    target = directory or repo_path("results", "signals", f"seed{seed}")
    local: dict[tuple, dict] = {}
    llm: dict[tuple, dict] = {}
    for path in sorted(target.glob("*.jsonl")):
        bucket = llm if ".llm." in path.name else local
        for row in _cells(path):
            bucket[(row["corpus_id"], row["query_id"], row["doc_id"])] = row

    joined: list[dict] = []
    misses = 0
    for key, row in local.items():
        merged = dict(row)
        other = llm.get(key)
        if other is None:
            misses += 1
            other = {}
        for name in LLM_COLUMNS:
            merged[name] = float(other.get(name, 0.0))
        for name in LLM_EXTRAS:
            merged[name] = other.get(name, 0 if name in ("s3_unknown", "call_failed") else "")
        if not other:
            merged["s3_unknown"] = 1
        merged["miss"] = 0 if other else 1
        joined.append(merged)

    if misses:
        joined[0]["_misses"] = misses
    return joined


def signal_columns(names: list[str]) -> list[str]:
    """Map ablation names to table columns, preserving order."""
    return [ABLATION_COLUMNS[name][1] for name in names]


def hard_subset(rows: list[dict]) -> list[dict]:
    """Poison plus the clean candidates that actually answered the question."""
    return [
        row
        for row in rows
        if int(row["is_attack_for_query"]) == 1
        or (row["label_class"] == "clean" and not int(row.get("s3_unknown", 1)))
    ]
