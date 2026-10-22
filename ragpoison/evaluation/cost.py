"""Aggregate the call ledger by query.

The acceptance criterion asks for calls, tokens, latency and cache-hit rate *per
query*, and the ledger only records them per call. This module reconstructs the
per-query view from the ``tag`` field that each stage attaches.

Not every stage is attributable: the poison-validity audit tags a passage rather than
a pool, so those calls are counted and reported separately instead of being quietly
attributed to the wrong query. Losing a few percent to "unattributed" is better than a
table that silently sums the wrong rows.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict

from ..datagen.corpora import iter_query_corpora
from ..datagen.loader import load_bundle
from ..paths import path as repo_path, rel


def _corpus_map(seed: int) -> dict[str, dict]:
    """corpus_id -> the query, family and injection it belongs to."""
    bundle = load_bundle(seed)
    queries = {row["query_id"]: row for row in bundle.queries}
    out: dict[str, dict] = {}
    # iter_query_corpora yields the same corpus ids the stages tag their calls with;
    # bundle.corpora is the twelve shared corpora and does not carry a query per row.
    for corpus in iter_query_corpora(bundle):
        query = queries.get(corpus["query_id"])
        if query is None:
            continue
        out[corpus["corpus_id"]] = {
            "query_id": query["query_id"],
            "fact_id": query["fact_id"],
            "split": query["split"],
            "attack_family": corpus["attack_family"],
            "injection": int(corpus["injection"]),
        }
    return out


def per_query(seed: int = 42) -> dict:
    """Return per-query totals and the unattributed remainder.

    The result is deliberately **seed-agnostic**, and that is a property of the data
    rather than a choice: corpus ids such as ``ATK1_PoisonedRAG_inj1_q_0001`` are
    identical in both knowledge bases, and the ledger records no seed, so there is no
    way to attribute a call to seed 42 rather than seed 43. Asking for either seed
    therefore returns the same pooled table. Reporting one table with that stated is
    honest; reporting two identical tables labelled "seed 42" and "seed 43" would not
    be.

    ``seed`` only selects which generated bundle is used to resolve a corpus id to its
    query, and that mapping is the same in both.
    """
    ledger = repo_path("logs", "llm_calls.jsonl")
    if not ledger.exists():
        return {"rows": [], "unattributed": 0, "total": 0}
    mapping = _corpus_map(seed)

    totals: dict[str, dict] = defaultdict(
        lambda: {
            "calls": 0,
            "paid_calls": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "latency_ms": 0,
            "pools": set(),
        }
    )
    unattributed = 0
    total = 0
    with ledger.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            total += 1
            corpus_id = None
            tag = row.get("tag")
            if isinstance(tag, dict):
                corpus_id = tag.get("corpus_id")
            info = mapping.get(corpus_id) if corpus_id else None
            if info is None:
                unattributed += 1
                continue
            bucket = totals[info["query_id"]]
            bucket["calls"] += 1
            if not row.get("cache_hit"):
                bucket["paid_calls"] += 1
            bucket["tokens_in"] += int(row.get("prompt_tokens") or 0)
            bucket["tokens_out"] += int(row.get("completion_tokens") or 0)
            bucket["latency_ms"] += int(row.get("latency_ms") or 0)
            bucket["pools"].add(corpus_id)
            bucket.setdefault("families", set()).add(info["attack_family"])
            bucket["split"] = info["split"]

    rows = []
    for query_id in sorted(totals):
        bucket = totals[query_id]
        calls = bucket["calls"] or 1
        rows.append(
            {
                "seed": seed,
                "query_id": query_id,
                "split": bucket.get("split", ""),
                "pools": len(bucket["pools"]),
                "calls": bucket["calls"],
                "paid_calls": bucket["paid_calls"],
                "cache_hit_rate": round(1.0 - bucket["paid_calls"] / calls, 4),
                "tokens_in": bucket["tokens_in"],
                "tokens_out": bucket["tokens_out"],
                "latency_ms": bucket["latency_ms"],
                "mean_latency_ms": round(bucket["latency_ms"] / calls, 1),
            }
        )
    return {"rows": rows, "unattributed": unattributed, "total": total}


def write_csv(seed: int = 42) -> str:
    payload = per_query(seed)
    path = repo_path("results", "cost_per_query.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "query_id",
        "split",
        "pools",
        "calls",
        "paid_calls",
        "cache_hit_rate",
        "tokens_in",
        "tokens_out",
        "latency_ms",
        "mean_latency_ms",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in payload["rows"]:
            writer.writerow({key: row.get(key) for key in fields})
    payload["path"] = rel(path)
    return payload
