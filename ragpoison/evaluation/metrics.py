"""Detection metrics.

Rank-based AUC is implemented here rather than imported so that tie handling is
explicit and the result is identical on every platform: every signal scores many
candidates at exactly 0 or exactly 1, so ties are the normal case, not an edge
case.

The rest of the metrics needed by the report arrive in WP6 together with the
per-cell aggregation; only what a pilot run actually reports is here.
"""

from __future__ import annotations


def roc_auc(scores: list[float], labels: list[int]) -> float | None:
    """Probability that a positive outranks a negative, ties counted as a half.

    Returns ``None`` when one class is absent, which is the honest answer rather
    than an arbitrary 0.5.
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must be the same length")
    positives = sum(1 for label in labels if label)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None

    pairs = sorted(zip(scores, labels), key=lambda pair: pair[0])
    ranks = [0.0] * len(pairs)
    index = 0
    while index < len(pairs):
        end = index
        while end + 1 < len(pairs) and pairs[end + 1][0] == pairs[index][0]:
            end += 1
        average_rank = (index + end + 2) / 2.0  # 1-based average rank for the tie group
        for position in range(index, end + 1):
            ranks[position] = average_rank
        index = end + 1

    rank_sum = sum(rank for rank, (_score, label) in zip(ranks, pairs) if label)
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[len(ordered) // 2])
