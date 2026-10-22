"""Statistics. Everything is resampled over pools, never over rows.

Row-level resampling would be wrong here and would flatter every confidence
interval: the fifteen candidates of one pool are not independent (they come from one
query), and at injection size three the three poisoned passages are near-duplicates
of each other, so a row-level bootstrap would pretend there are three times as many
independent attack samples as there are.

A "pool" is one ``(corpus_id, query_id)`` pair, which is one query under one attack
family at one injection size.

McNemar's test is used for paired comparisons: each pool either has all of its
poisoned passages removed or it does not, and two configurations are compared on
the pools where they disagree. Exact binomial tails are computed directly rather
than through a chi-square approximation, because the number of discordant pools is
often small.
"""

from __future__ import annotations

import math
import random
from collections import Counter


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = int(math.floor(position))
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def cluster_bootstrap(
    per_pool: dict[str, float],
    *,
    reps: int = 1000,
    seed: int = 20251021,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Mean of a per-pool quantity with a percentile interval.

    Returns ``(observed_mean, low, high)``. The resampling unit is the pool.
    """
    keys = list(per_pool)
    if not keys:
        return 0.0, 0.0, 0.0
    values = [per_pool[key] for key in keys]
    observed = sum(values) / len(values)
    if len(keys) == 1:
        return observed, observed, observed
    generator = random.Random(seed)
    means: list[float] = []
    n = len(keys)
    for _ in range(reps):
        total = 0.0
        for _ in range(n):
            total += values[generator.randrange(n)]
        means.append(total / n)
    return observed, percentile(means, alpha / 2), percentile(means, 1 - alpha / 2)


def _binomial_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial tail probability for p=0.5."""
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1))
    return min(1.0, 2 * tail / (2**n))


def mcnemar(
    reference: dict[str, int],
    candidate: dict[str, int],
) -> dict:
    """Paired comparison over pools.

    ``reference`` and ``candidate`` map pool key to 1/0 ("defence succeeded on this
    pool"). Pools present in only one mapping are ignored, and the count of such
    pools is reported.
    """
    shared = [key for key in reference if key in candidate]
    b = sum(1 for key in shared if reference[key] == 1 and candidate[key] == 0)
    c = sum(1 for key in shared if reference[key] == 0 and candidate[key] == 1)
    n = b + c
    p_value = _binomial_two_sided(min(b, c), n) if n else 1.0
    return {
        "n_pairs": len(shared),
        "only_reference": b,
        "only_candidate": c,
        "discordant": n,
        "p_value": p_value,
        "significant": p_value < 0.05,
    }


def summarise_counts(counter: Counter) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}
