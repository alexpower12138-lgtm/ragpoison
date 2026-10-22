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


def paired_risk_difference(wins: int, losses: int, n: int) -> tuple[float, float, float]:
    """Difference in proportions for a paired comparison, with a Wald interval.

    ``wins`` counts pairs where the candidate succeeded and the reference did not,
    ``losses`` the reverse. The estimate is ``(wins - losses) / n``, which is the
    same number as ``p_candidate - p_reference`` but computed from the discordant
    pairs, so it carries the pairing. The variance is Agresti's:

        (wins + losses - (wins - losses)^2 / n) / n^2

    This is the effect size the report leads with, because it is in the same unit as
    ``clean_rate`` and shares its 2x2 table with McNemar, so the test and the effect
    size can never disagree about what was compared.

    A Wald interval is poor when ``wins + losses`` is small; the caller should say so
    where that applies rather than trusting the interval blindly.
    """
    if n <= 0:
        return 0.0, 0.0, 0.0
    difference = (wins - losses) / n
    variance = (wins + losses - (wins - losses) ** 2 / n) / (n * n)
    if variance <= 0:
        return difference, difference, difference
    half = 1.959963984540054 * math.sqrt(variance)
    return difference, difference - half, difference + half


def cohens_h(p1: float, p2: float) -> float:
    """Standardised difference between two proportions.

    Reported as a secondary measure only. It is defined for *independent* samples, so
    it discards the pairing and will usually understate the effect here.
    """
    return 2 * math.asin(math.sqrt(min(max(p1, 0.0), 1.0))) - 2 * math.asin(
        math.sqrt(min(max(p2, 0.0), 1.0))
    )


def wilcoxon_signed_rank(pairs: list[tuple[float, float]]) -> dict:
    """Exact two-sided Wilcoxon signed-rank test for a small number of pairs.

    Written here rather than imported so the result is exact and identical on every
    machine: with the twelve attack-by-injection cells this project compares, the
    exact null distribution is 2**12 = 4096 sign assignments and there is no reason to
    approximate it. Ties in the absolute differences get average ranks, and pairs with
    a zero difference are dropped, which is the standard convention.

    The caller must report ``n``: with n = 12 the smallest attainable two-sided
    p-value is about 0.0005, so significance at that size is easy and should not be
    over-read.
    """
    differences = [first - second for first, second in pairs if first != second]
    n = len(differences)
    result = {
        "n": n,
        "n_input": len(pairs),
        "w_plus": 0.0,
        "w_minus": 0.0,
        "p_value": 1.0,
        "method": "exact",
    }
    if n == 0:
        return result

    ordered = sorted(range(n), key=lambda index: abs(differences[index]))
    ranks = [0.0] * n
    position = 0
    while position < n:
        end = position
        while end + 1 < n and abs(differences[ordered[end + 1]]) == abs(differences[ordered[position]]):
            end += 1
        average = (position + end + 2) / 2.0
        for cursor in range(position, end + 1):
            ranks[ordered[cursor]] = average
        position = end + 1

    w_plus = sum(rank for rank, diff in zip(ranks, differences) if diff > 0)
    total = sum(ranks)
    result["w_plus"] = w_plus
    result["w_minus"] = total - w_plus

    if n <= 16:
        observed = abs(w_plus - total / 2.0)
        extreme = 0
        for mask in range(1 << n):
            plus = 0.0
            for index in range(n):
                if mask >> index & 1:
                    plus += ranks[index]
            if abs(plus - total / 2.0) >= observed - 1e-9:
                extreme += 1
        result["p_value"] = extreme / float(1 << n)
    else:  # pragma: no cover - not reached by the current cell counts
        mean = total / 2.0
        variance = n * (n + 1) * (2 * n + 1) / 24.0
        if variance <= 0:
            return result
        z = (abs(w_plus - mean) - 0.5) / math.sqrt(variance)
        result["p_value"] = math.erfc(z / math.sqrt(2.0))
        result["method"] = "normal approximation"
    return result


def summarise_counts(counter: Counter) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}
