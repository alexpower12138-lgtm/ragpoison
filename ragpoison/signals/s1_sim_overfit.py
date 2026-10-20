"""S1 - similarity over-fit.

A poisoned passage is built to be retrieved: it repeats the question's wording, or
stuffs the question's keywords into a short tail. That makes it score *higher* than
the ordinary passages around it, which is the signal:

    z_i     = (b_i - median(b)) / (max(b) - median(b))      in [0, 1]
    s1_i    = sigmoid((z_i - 0.5) / 0.15)

Only "abnormally close" is penalised; a candidate at or below the median scores
near zero. This is the same intuition as the text-similarity filter in RAGuard
[7], and it is deliberately *not* the naive "high rank means relevant means
clean" reading, which is backwards for this threat model.

Known bias, and the reason ablation cell C1f flips the sign: the top-ranked
candidate is almost always scored as the most suspicious, which is correct when
the top-ranked candidate is the attack and wrong when it is the genuine evidence.
That trade-off is measured, not assumed.
"""

from __future__ import annotations

import math

from .base import SignalContext, SignalResult, clip01, register

NAME = "s1"

#: Steepness of the transition around the midpoint, in units of z.
STEEPNESS = 0.15


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def compute(context: SignalContext) -> SignalResult:
    scores = [candidate.bm25 for candidate in context.candidates]
    if not scores:
        return SignalResult(NAME, {})

    ordered = sorted(scores)
    median = ordered[len(ordered) // 2]
    maximum = ordered[-1]
    span = maximum - median
    details: dict[str, float] = {"median": median, "max": maximum}

    out: dict[str, float] = {}
    for candidate in context.candidates:
        if span <= 1e-9:
            z = 0.0 if candidate.bm25 <= median else 1.0
        else:
            z = max(0.0, (candidate.bm25 - median) / span)
        out[candidate.doc_id] = clip01(_sigmoid((z - 0.5) / STEEPNESS))
        details[f"z:{candidate.doc_id}"] = round(z, 6)
    return SignalResult(NAME, out, details)


register(compute)
