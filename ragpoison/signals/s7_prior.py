"""S7 - retrieval prior.

This is **not a verdict**. A high BM25 score means "this passage matches the
query", which for a poisoned passage is exactly what the attacker engineered, and
for a genuine passage is exactly what the retriever is for. The two cases are
indistinguishable from the score alone.

So S7 only appears as a multiplier in the weighted fusion:

    w_i  = prior_weight_scale + (1 - prior_weight_scale) * b_i / max(b)
    S_i  = w_i * sum_k alpha_k s_k_i

and never enters the mean, the threshold or the logistic model. Ablation cells
C1/C1f quantify what happens when the prior is mistaken for a verdict.
"""

from __future__ import annotations

from .base import SignalContext, SignalResult, clip01, register

NAME = "s7"


def compute(context: SignalContext) -> SignalResult:
    scores = [candidate.bm25 for candidate in context.candidates]
    if not scores:
        return SignalResult(NAME, {})
    scale = float(context.config.get("fusion.prior_weight_scale", 0.5))
    scale = min(max(scale, 0.0), 1.0)
    maximum = max(scores) or 1.0

    out: dict[str, float] = {}
    for candidate in context.candidates:
        normalized = candidate.bm25 / maximum if maximum > 0 else 0.0
        out[candidate.doc_id] = clip01(scale + (1.0 - scale) * normalized)
    return SignalResult(NAME, out, {"scale": scale})


register(compute)
