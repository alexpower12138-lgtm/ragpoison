"""The filtering policy: what the defence actually does to one retrieved pool.

Given the fused score and a threshold, produce the context that would be handed to
the generator:

1. drop every candidate scoring above the threshold;
2. if at least ``k`` candidates survive, keep the best ``k`` by the original BM25
   order (the retriever's ordering is not re-derived from the suspicion score);
3. if fewer survive, refill with the *least* suspicious of the dropped ones, and
   count that: a defence that has to put poison back is not a defence;
4. if everything was dropped, the pool is empty and the generator should be asked
   with no context, which the measurement treats as an abstention by design.

The policy is deliberately the simplest thing that can work. Anything cleverer
(thresholds per query, learned cut-offs) would need its own calibration and would
make the ablation table harder to attribute.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..signals.load import row_key

THRESHOLD_SLACK = 1e-12


@dataclass
class FilteredPool:
    corpus_id: str
    query_id: str
    attack_family: str
    injection: int
    kept: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    refilled: list[str] = field(default_factory=list)
    abstained: bool = False
    atr_before: float = 0.0
    atr_after: float = 0.0
    poison_before: int = 0
    poison_after: int = 0
    all_poison_removed: bool = True
    filtered_ratio: float = 0.0


def filter_pool(
    rows: list[dict],
    scores: dict[tuple, float],
    threshold: float,
    k: int,
) -> FilteredPool:
    """Apply the policy to one instance. ``rows`` must be sorted by rank."""
    ordered = sorted(rows, key=lambda row: row["rank"])
    if not ordered:
        raise ValueError("empty pool")
    head = ordered[0]
    pool = FilteredPool(
        corpus_id=head["corpus_id"],
        query_id=head["query_id"],
        attack_family=head["attack_family"],
        injection=int(head["injection"]),
    )
    poison_ids = {row["doc_id"] for row in ordered if int(row["is_attack_for_query"]) == 1}
    pool.poison_before = len(poison_ids)
    pool.atr_before = pool.poison_before / len(ordered)

    survivors = [row for row in ordered if scores[row_key(row)] <= threshold + THRESHOLD_SLACK]
    removed = [row for row in ordered if scores[row_key(row)] > threshold + THRESHOLD_SLACK]
    pool.filtered_ratio = len(removed) / len(ordered)

    if not survivors:
        pool.abstained = True
        pool.dropped = [row["doc_id"] for row in ordered]
        pool.kept = []
        pool.atr_after = 0.0
        pool.poison_after = 0
        pool.all_poison_removed = True
        return pool

    kept = survivors[:k]
    refilled: list[dict] = []
    if len(kept) < k:
        spare = sorted(removed, key=lambda row: scores[row_key(row)])
        refilled = spare[: k - len(kept)]
        kept = kept + refilled

    kept_ids = {row["doc_id"] for row in kept}
    pool.kept = [row["doc_id"] for row in kept]
    pool.refilled = [row["doc_id"] for row in refilled]
    pool.dropped = [row["doc_id"] for row in ordered if row["doc_id"] not in kept_ids]
    pool.poison_after = len(kept_ids & poison_ids)
    pool.atr_after = pool.poison_after / len(kept_ids) if kept_ids else 0.0
    pool.all_poison_removed = pool.poison_after == 0
    return pool


def empty_pool_keeps_everything(rows: list[dict], k: int) -> FilteredPool:
    """The no-defence case (C0): the top-k pool passes through untouched."""
    ordered = sorted(rows, key=lambda row: row["rank"])
    head = ordered[0]
    kept = ordered[:k]
    poison_ids = {row["doc_id"] for row in ordered if int(row["is_attack_for_query"]) == 1}
    kept_ids = {row["doc_id"] for row in kept}
    poison_after = len(kept_ids & poison_ids)
    return FilteredPool(
        corpus_id=head["corpus_id"],
        query_id=head["query_id"],
        attack_family=head["attack_family"],
        injection=int(head["injection"]),
        kept=[row["doc_id"] for row in kept],
        dropped=[],
        refilled=[],
        abstained=False,
        atr_before=len(poison_ids) / len(ordered),
        atr_after=poison_after / len(kept_ids) if kept_ids else 0.0,
        poison_before=len(poison_ids),
        poison_after=poison_after,
        all_poison_removed=poison_after == 0,
        filtered_ratio=0.0,
    )
