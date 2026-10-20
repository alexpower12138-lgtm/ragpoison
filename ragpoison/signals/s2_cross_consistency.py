"""S2 - cross-passage consistency.

A retrieval pool for one query usually contains several passages about the same
neighbourhood of facts. A passage that looks like nothing else in the pool is worth
a second look:

    avg_sim_i = mean_{j != i} cosine_tfidf(d_i, d_j)
    s2_i      = clip(0.5 - (avg_sim_i - mu) / (6 * sigma), 0, 1)

``mu`` and ``sigma`` come from benign pools built on the **calibration split only**,
so the signal never sees the test data it is scored on.

Limits, stated up front because they matter when reading the results: this signal
detects *topical outliers*, not fine-grained contradictions inside the same topic.
A poisoned passage that stays on topic (ATK2, ATK5, ATK6) will look consistent, and
the signal should be flat there. ATK4's shuffled keyword tail is the case where it
should fire.
"""

from __future__ import annotations

from .base import SignalContext, SignalResult, clip01, register

NAME = "s2"


def compute(context: SignalContext) -> SignalResult:
    ids = context.ids()
    if len(ids) < 2:
        return SignalResult(NAME, {doc_id: 0.0 for doc_id in ids})

    means = context.tfidf.mean_similarity_to_others(ids)
    stats = context.calibration.get("s2", {})
    mu = float(stats.get("mu", 0.0))
    sigma = float(stats.get("sigma", 1.0)) or 1.0

    details: dict[str, float] = {"mu": mu, "sigma": sigma}
    out: dict[str, float] = {}
    for doc_id, avg_sim in means.items():
        out[doc_id] = clip01(0.5 - (avg_sim - mu) / (6.0 * sigma))
        details[f"avg_sim:{doc_id}"] = round(avg_sim, 6)
    return SignalResult(NAME, out, details)


register(compute)
