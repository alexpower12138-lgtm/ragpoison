"""Metric definitions, pinned in one place.

Candidate level (every retrieved candidate is one observation):
    precision, recall, F1, FPR, FNR - positive class is ``is_attack_for_query``.
    micro pools all candidates; macro averages the six attack families first.

Pool level (one query under one attack family at one injection size):
    ``clean_rate`` - the fraction of pools where no poisoned passage survives into
    the context that would be handed to the generator. This is the number the
    project cares about, because it is the only one that maps directly onto "the
    attack no longer reaches the model".

The reported ``clean_rate`` is computed only over pools where the attack actually
reached the *undefended* top-k. Pools where the poison never made the context are
counted separately and excluded: a defence cannot be credited for an attack that
was already outside the window, and including them would make every configuration
look better by a constant.
"""

from __future__ import annotations

from collections import Counter

from ..fusion.policy import FilteredPool
from ..signals.load import row_key


def pool_index(rows: list[dict]) -> dict[str, list[dict]]:
    """Group rows into pools, each sorted by rank."""
    pools: dict[str, list[dict]] = {}
    for row in rows:
        key = f"{row['corpus_id']}|{row['query_id']}"
        pools.setdefault(key, []).append(row)
    for subset in pools.values():
        subset.sort(key=lambda row: row["rank"])
    return pools


def baseline_poison_in_topk(pools: dict[str, list[dict]], k: int) -> dict[str, int]:
    """Poisoned passages inside the undefended top-k, per pool."""
    out: dict[str, int] = {}
    for key, rows in pools.items():
        kept = rows[:k]
        out[key] = sum(1 for row in kept if int(row["is_attack_for_query"]) == 1)
    return out


def candidate_metrics(rows: list[dict], scores: dict[tuple, float], threshold: float) -> dict:
    tp = fp = fn = tn = 0
    for row in rows:
        positive = int(row["is_attack_for_query"]) == 1
        flagged = scores[row_key(row)] > threshold
        if positive and flagged:
            tp += 1
        elif positive:
            fn += 1
        elif flagged:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "fnr": fn / (fn + tp) if fn + tp else 0.0,
    }


def macro_f1(pools_by_family: dict[str, list[dict]], scores: dict[tuple, float], threshold: float) -> float:
    values = [
        candidate_metrics(rows, scores, threshold)["f1"]
        for rows in pools_by_family.values()
        if rows
    ]
    return sum(values) / len(values) if values else 0.0


def summarise(
    filtered: list[FilteredPool],
    *,
    universe: set[str],
    candidate: dict,
    macro: float,
    auc: float | None,
    auc_hard: float | None,
    threshold: float,
    threshold_f1: float,
) -> dict:
    """Roll a configuration's pools and candidates into one result record."""
    pool_key = lambda pool: f"{pool.corpus_id}|{pool.query_id}"
    in_universe = [pool for pool in filtered if pool_key(pool) in universe]
    successes = Counter(
        {pool_key(pool): int(bool(pool.all_poison_removed)) for pool in in_universe}
    )
    clean_rate = (
        sum(successes.values()) / len(in_universe) if in_universe else 0.0
    )
    return {
        "n_pools": len(filtered),
        "n_pools_scored": len(in_universe),
        "n_pools_attack_outside_window": len(filtered) - len(in_universe),
        "clean_rate": clean_rate,
        "atr_before": _mean(pool.atr_before for pool in filtered),
        "atr_after": _mean(pool.atr_after for pool in filtered),
        "filtered_ratio": _mean(pool.filtered_ratio for pool in filtered),
        "refills": sum(1 for pool in filtered if pool.refilled),
        "abstentions": sum(1 for pool in filtered if pool.abstained),
        "tau": threshold,
        "tau_f1": threshold_f1,
        "precision": candidate["precision"],
        "recall": candidate["recall"],
        "f1": candidate["f1"],
        "f1_macro": macro,
        "fpr": candidate["fpr"],
        "fnr": candidate["fnr"],
        "tp": candidate["tp"],
        "fp": candidate["fp"],
        "fn": candidate["fn"],
        "tn": candidate["tn"],
        "auc": auc,
        "auc_hard": auc_hard,
        "_successes": {key: value for key, value in successes.items()},
    }


def _mean(values) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def family_pools(pools: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Flatten pools into per-attack-family row lists, for macro averaging."""
    out: dict[str, list[dict]] = {}
    for rows in pools.values():
        family = rows[0]["attack_family"]
        out.setdefault(family, []).extend(rows)
    return out


def by_injection(pools: dict[str, list[dict]]) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for key, rows in pools.items():
        out.setdefault(int(rows[0]["injection"]), []).append(key)
    return out
