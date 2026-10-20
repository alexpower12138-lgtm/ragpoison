"""Stage R of the attack-validity audit: is a poisoned passage actually reachable?

This is the free, offline half of WP2 and the gate everything else depends on. If
an attack cannot place its passage inside the detection pool, the cell is empty
and no downstream analysis can rescue it.

Two views are reported, matching the two research questions:

* **query-time** (RQ1): the knowledge base a query actually sees, i.e. the base
  pool plus only that query's own poisoned passages (see
  ``ragpoison/datagen/corpora.py``). This is the view every detection and
  generation metric uses.
* **corpus-time** (RQ2): the shared corpora on disk, where the base contains one
  hundred poisoned chunks. This measures the situation the periodic corpus audit
  exists for, and quantifies how badly a retrieval pool is contaminated there.

Outputs
-------
``reports/audit_retrieval_seed<N>.md``   human-readable tables
``results/audit_retrieval_seed<N>.json`` machine-readable per-cell rows
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..datagen.corpora import iter_query_corpora
from ..datagen.loader import Bundle, load_bundle
from ..paths import path as repo_path, rel
from ..retrieval.bm25 import BM25

#: Gate R thresholds from the specification: a canonical attack has to reach the
#: pool almost always, a stealth attack is allowed to be slightly weaker because
#: it never echoes the question.
GATE_R_CANONICAL = 0.85
GATE_R_STEALTH = 0.60

CANONICAL_FAMILIES = (
    "ATK1_PoisonedRAG",
    "ATK2_CorruptRAG_AS",
    "ATK3_InstructionInjection",
    "ATK4_DIGA_Stuffing",
)


@dataclass
class QueryCorpusResult:
    """Aggregated result for one (attack family, injection) under the RQ1 view."""

    attack_family: str
    injection: int
    n_queries: int = 0
    poison_in_pool: int = 0
    poison_in_top5: int = 0
    poison_ranks: list[int] = field(default_factory=list)
    clean_in_pool: int = 0
    clean_ranks: list[int] = field(default_factory=list)
    cells: list[dict] = field(default_factory=list)

    @property
    def n_poison(self) -> int:
        return self.n_queries * self.injection

    @property
    def pool_rate(self) -> float:
        return self.poison_in_pool / self.n_poison if self.n_poison else 0.0

    @property
    def top5_rate(self) -> float:
        return self.poison_in_top5 / self.n_poison if self.n_poison else 0.0

    @property
    def clean_pool_rate(self) -> float:
        return self.clean_in_pool / self.n_queries if self.n_queries else 0.0

    @property
    def mean_clean_rank(self) -> float:
        return sum(self.clean_ranks) / len(self.clean_ranks) if self.clean_ranks else 0.0

    @property
    def worst_clean_rank(self) -> int:
        return max(self.clean_ranks) if self.clean_ranks else 0

    @property
    def mean_poison_rank(self) -> float:
        return sum(self.poison_ranks) / len(self.poison_ranks) if self.poison_ranks else 0.0

    @property
    def worst_poison_rank(self) -> int:
        return max(self.poison_ranks) if self.poison_ranks else 0


@dataclass
class SharedCorpusResult:
    """Aggregated result for one shared corpus (the RQ2 view)."""

    corpus_id: str
    attack_family: str
    injection: int
    n_queries: int = 0
    poison_per_pool: float = 0.0
    foreign_poison_per_pool: float = 0.0
    clean_in_pool: int = 0

    @property
    def clean_pool_rate(self) -> float:
        return self.clean_in_pool / self.n_queries if self.n_queries else 0.0


def _pool_size() -> int:
    try:
        from ..config import load_config

        return int(load_config("default.yaml").get("retrieval.top_n", 15))
    except Exception:  # noqa: BLE001 - the audit must run without configs too
        return 15


def run(seed: int = 42, top_n: int | None = None, write: bool = True) -> dict:
    """Rank every corpus in both views and report reachability."""
    bundle: Bundle = load_bundle(seed)
    pool = top_n or _pool_size()
    top_k = 5
    texts = bundle.texts()
    queries = {row["query_id"]: row for row in bundle.queries}
    clean_id = {row["query_id"]: bundle.clean_id_for_fact(row["fact_id"]) for row in bundle.queries}

    # ---------------------------------------------------------------- RQ1 view
    grouped: dict[tuple[str, int], QueryCorpusResult] = {}
    for corpus in iter_query_corpora(bundle):
        key = (corpus["attack_family"], int(corpus["injection"]))
        result = grouped.get(key)
        if result is None:
            result = QueryCorpusResult(*key)
            grouped[key] = result

        documents = [(chunk_id, texts[chunk_id]) for chunk_id in corpus["chunk_ids"]]
        index = BM25().fit(documents)
        query_id = corpus["query_id"]
        query = queries[query_id]
        ordered = index.ranked_ids(query["question"])
        ranks = {doc_id: position for position, (doc_id, _score) in enumerate(ordered, start=1)}

        own_ranks = [ranks[chunk_id] for chunk_id in corpus["poison_chunk_ids"]]
        in_pool = sum(1 for rank in own_ranks if rank <= pool)
        in_top5 = sum(1 for rank in own_ranks if rank <= top_k)
        clean_rank = ranks.get(clean_id[query_id], 0)

        result.n_queries += 1
        result.poison_in_pool += in_pool
        result.poison_in_top5 += in_top5
        result.poison_ranks.extend(own_ranks)
        result.clean_ranks.append(clean_rank)
        result.clean_in_pool += 1 if 0 < clean_rank <= pool else 0
        result.cells.append(
            {
                "corpus_id": corpus["corpus_id"],
                "attack_family": result.attack_family,
                "injection": result.injection,
                "query_id": query_id,
                "fact_id": query["fact_id"],
                "n_chunks": corpus["n_chunks"],
                "clean_rank": clean_rank,
                "own_poison_ranks": own_ranks,
                "own_poison_in_pool": in_pool,
                "own_poison_in_top5": in_top5,
            }
        )

    query_results = [grouped[key] for key in sorted(grouped)]

    # ---------------------------------------------------------------- RQ2 view
    shared_results: list[SharedCorpusResult] = []
    for corpus_id in sorted(bundle.corpora):
        corpus = bundle.corpora[corpus_id]
        documents = [(chunk_id, texts[chunk_id]) for chunk_id in corpus["chunk_ids"]]
        index = BM25().fit(documents)
        poison_ids = set(corpus["poison_chunk_ids"])
        owner = corpus["poison_owner"]
        result = SharedCorpusResult(
            corpus_id=corpus_id,
            attack_family=corpus["attack_family"],
            injection=int(corpus["injection"]),
        )
        total_poison = 0
        total_foreign = 0
        for query in bundle.queries:
            query_id = query["query_id"]
            ordered = index.ranked_ids(query["question"])
            pool_ids = [doc_id for doc_id, _score in ordered[:pool]]
            total_poison += sum(1 for doc_id in pool_ids if doc_id in poison_ids)
            total_foreign += sum(
                1 for doc_id in pool_ids if doc_id in poison_ids and owner.get(doc_id) != query_id
            )
            clean_rank = next(
                (position for position, (doc_id, _score) in enumerate(ordered, start=1)
                 if doc_id == clean_id[query_id]),
                0,
            )
            result.n_queries += 1
            result.clean_in_pool += 1 if 0 < clean_rank <= pool else 0
        result.poison_per_pool = total_poison / max(result.n_queries, 1)
        result.foreign_poison_per_pool = total_foreign / max(result.n_queries, 1)
        shared_results.append(result)

    gate_ok, gate_detail = _gate_r(query_results)
    payload = {
        "seed": seed,
        "pool_size": pool,
        "top_k": top_k,
        "generator_version": bundle.manifest.get("generator_version"),
        "manifest_sha256": bundle.manifest.get("sha256", {}),
        "gate_r": {"passed": gate_ok, "detail": gate_detail},
        "query_time": [_query_payload(row) for row in query_results],
        "corpus_time": [_shared_payload(row) for row in shared_results],
    }

    if write:
        json_path = repo_path("results", f"audit_retrieval_seed{seed}.json")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
        )
        md_path = repo_path("reports", f"audit_retrieval_seed{seed}.md")
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(
            render_markdown(bundle, query_results, shared_results, pool, gate_ok, gate_detail),
            encoding="utf-8",
            newline="\n",
        )
        payload["_written"] = [rel(json_path), rel(md_path)]

    payload["_query_results"] = query_results
    payload["_shared_results"] = shared_results
    payload["_bundle"] = bundle
    return payload


def _query_payload(row: QueryCorpusResult) -> dict:
    return {
        "attack_family": row.attack_family,
        "injection": row.injection,
        "n_queries": row.n_queries,
        "n_poison": row.n_poison,
        "poison_in_pool_rate": row.pool_rate,
        "poison_in_top5_rate": row.top5_rate,
        "clean_in_pool_rate": row.clean_pool_rate,
        "mean_poison_rank": row.mean_poison_rank,
        "worst_poison_rank": row.worst_poison_rank,
        "mean_clean_rank": row.mean_clean_rank,
        "worst_clean_rank": row.worst_clean_rank,
        "cells": row.cells,
    }


def _shared_payload(row: SharedCorpusResult) -> dict:
    return {
        "corpus_id": row.corpus_id,
        "attack_family": row.attack_family,
        "injection": row.injection,
        "n_queries": row.n_queries,
        "poison_per_pool": row.poison_per_pool,
        "foreign_poison_per_pool": row.foreign_poison_per_pool,
        "clean_in_pool_rate": row.clean_pool_rate,
    }


def _gate_r(results: list[QueryCorpusResult]) -> tuple[bool, str]:
    """Gate R: every family must reach the pool in at least its threshold."""
    thresholds: dict[str, float] = {}
    for row in results:
        threshold = GATE_R_CANONICAL if row.attack_family in CANONICAL_FAMILIES else GATE_R_STEALTH
        thresholds[row.attack_family] = threshold

    failures: list[str] = []
    canonical_worst = 1.0
    stealth_worst = 1.0
    for family, threshold in sorted(thresholds.items()):
        rates = [row.pool_rate for row in results if row.attack_family == family]
        worst = min(rates) if rates else 0.0
        if family in CANONICAL_FAMILIES:
            canonical_worst = min(canonical_worst, worst)
        else:
            stealth_worst = min(stealth_worst, worst)
        if worst < threshold:
            failures.append(f"{family}={worst:.2%} < {threshold:.0%}")

    detail = (
        f"canonical worst {canonical_worst:.2%} (threshold {GATE_R_CANONICAL:.0%}), "
        f"stealth worst {stealth_worst:.2%} (threshold {GATE_R_STEALTH:.0%})"
    )
    if failures:
        return False, detail + "; below threshold: " + ", ".join(failures)
    return True, detail


def render_markdown(
    bundle: Bundle,
    query_results: list[QueryCorpusResult],
    shared_results: list[SharedCorpusResult],
    pool: int,
    gate_ok: bool,
    gate_detail: str,
) -> str:
    manifest_sha = bundle.manifest.get("sha256", {})
    lines: list[str] = []
    lines.append(f"# Retrieval audit (stage R) - seed {bundle.seed}")
    lines.append("")
    lines.append("Offline audit: no language model is called, so this stage costs nothing.")
    lines.append("")
    lines.append(f"- detection pool: top-{pool}")
    lines.append(f"- generator version: {bundle.manifest.get('generator_version')}")
    lines.append(
        f"- poisoned_passages.json sha256: `{manifest_sha.get('poisoned_passages.json', '')}`"
    )
    lines.append(f"- queries / facts: {len(bundle.queries)} / {len(bundle.facts)}")
    lines.append("")
    lines.append(f"**Gate R: {'PASS' if gate_ok else 'FAIL'}** - {gate_detail}")
    lines.append("")
    lines.append("## View 1: query-time corpora (RQ1)")
    lines.append("")
    lines.append(
        "Knowledge base seen by one query: 100 clean passages + 200 distractors + only that "
        "query's own poisoned passages (301 or 303 chunks). This is the view every detection "
        "and generation metric uses."
    )
    lines.append("")
    lines.append(
        "| family | inj | poison in pool | poison in top-5 | mean poison rank | "
        "clean in pool | mean clean rank | worst clean rank |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in query_results:
        lines.append(
            f"| {row.attack_family} | {row.injection} | {row.pool_rate:.1%} | {row.top5_rate:.1%} "
            f"| {row.mean_poison_rank:.1f} | {row.clean_pool_rate:.1%} "
            f"| {row.mean_clean_rank:.1f} | {row.worst_clean_rank} |"
        )
    lines.append("")
    lines.append("## View 2: shared corpora (RQ2, the corpus-audit situation)")
    lines.append("")
    lines.append(
        "One knowledge base per attack family containing *all* 100 queries' poisoned passages "
        "(400 or 600 chunks). Poisoned text written for other queries fills the retrieval pool; "
        "this is precisely the situation the periodic corpus audit must handle, and the reason "
        "the two views are kept apart."
    )
    lines.append("")
    lines.append(
        "| corpus | family | inj | poison per pool | of which foreign | clean in pool |"
    )
    lines.append("|---|---|---|---|---|---|")
    for row in shared_results:
        lines.append(
            f"| {row.corpus_id} | {row.attack_family} | {row.injection} "
            f"| {row.poison_per_pool:.2f} / {pool} | {row.foreign_poison_per_pool:.2f} "
            f"| {row.clean_pool_rate:.1%} |"
        )
    lines.append("")
    lines.append("## Reading these numbers")
    lines.append("")
    lines.append(
        "- *poison in pool* is the share of targeted poisoned passages that reach the "
        f"top-{pool} detection pool. This is the precondition for every later cell."
    )
    lines.append(
        "- *poison in top-5* is the share that would enter the generator context with no "
        "defence at all. A high value means the attack is genuinely strong."
    )
    lines.append(
        "- *clean in pool* below 100% means the clean supporting passage was pushed out; those "
        "cells are excluded from later statistics and the exclusion rate is reported."
    )
    lines.append(
        "- In view 2, *of which foreign* counts poisoned passages written for a different "
        "query. They are genuinely poisoned text, so the corpus audit should flag them, but "
        "they are not attacks on the query being evaluated, which is why view 1 removes them."
    )
    lines.append("")
    return "\n".join(lines)


def summarize(payload: dict) -> str:
    """One-screen summary for the console."""
    query_results: list[QueryCorpusResult] = payload["_query_results"]
    shared_results: list[SharedCorpusResult] = payload["_shared_results"]
    gate = payload["gate_r"]
    lines = [
        f"pool size: top-{payload['pool_size']}",
        f"GATE R: {'PASS' if gate['passed'] else 'FAIL'}  ({gate['detail']})",
        "",
        "query-time corpora (RQ1)",
        f"{'family':<26}{'inj':>4}{'in pool':>10}{'in top5':>10}{'clean':>9}{'meanP':>8}",
    ]
    for row in query_results:
        lines.append(
            f"{row.attack_family:<26}{row.injection:>4}{row.pool_rate:>9.1%}"
            f"{row.top5_rate:>10.1%}{row.clean_pool_rate:>9.1%}{row.mean_poison_rank:>8.1f}"
        )
    lines.append("")
    lines.append("shared corpora (RQ2): poison per pool, of which foreign")
    for row in shared_results:
        lines.append(
            f"  {row.corpus_id:<32} {row.poison_per_pool:5.2f} / {payload['pool_size']}"
            f"   foreign {row.foreign_poison_per_pool:5.2f}"
        )
    if payload.get("_written"):
        lines.append("")
        for item in payload["_written"]:
            lines.append(f"wrote {item}")
    return "\n".join(lines)
