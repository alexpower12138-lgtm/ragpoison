"""S6 - pseudo-perplexity from a character n-gram model.

A character 3-gram language model is trained on the benign corpus only, and every
candidate is scored by its mean negative log-likelihood under that model:

    nll_i = -mean_t log P(c_t | c_{t-2}, c_{t-1})
    s6_i  = clip((nll_i - median) / (3 * MAD) / 2, 0, 1)

Median and MAD are used instead of mean and standard deviation because the whole
point is to be robust to a few very strange passages.

This signal is off by default (``detectors.enable_pseudo_ppl: false``). It should
fire on text whose *surface* is broken - ATK4's shuffled word order is the obvious
case - and be close to flat on fluent attacks. That is a hypothesis the ablation
tests, not an assumption.

Caveat for the report: the model is trained on the same benign passages it scores,
so benign text enjoys an in-sample advantage. With add-k smoothing over 300
passages the effect is small, but it means S6's absolute level is not a calibrated
probability and only its ranking should be read.
"""

from __future__ import annotations

import math
from collections import Counter

from .base import SignalContext, SignalResult, clip01, register

NAME = "s6"

ORDER = 2          # two characters of context
SMOOTHING = 1.0    # add-k


def _padded(text: str) -> str:
    return f"^{text.lower()}$"


def _nll(text: str, counts: dict[str, int], context_totals: dict[str, int], vocabulary: int) -> float:
    padded = _padded(text)
    if len(padded) <= ORDER:
        return 0.0
    total_nll = 0.0
    grams = 0
    for index in range(len(padded) - ORDER):
        gram = padded[index : index + ORDER + 1]
        context = gram[:-1]
        observed = counts.get(gram, 0)
        denominator = context_totals.get(context, 0) + SMOOTHING * vocabulary
        total_nll -= math.log((observed + SMOOTHING) / denominator)
        grams += 1
    return total_nll / grams if grams else 0.0


def compute(context: SignalContext) -> SignalResult:
    statistics = context.calibration.get("s6", {})
    if not statistics:
        return SignalResult(NAME, {doc_id: 0.0 for doc_id in context.ids()})

    counts: dict[str, int] = statistics["counts"]
    context_totals: dict[str, int] = statistics["context_totals"]
    vocabulary = int(statistics["vocabulary"]) or 1
    median = float(statistics["median"])
    mad = float(statistics["mad"]) or 1.0

    details: dict[str, float] = {"median": median, "mad": mad}
    out: dict[str, float] = {}
    for candidate in context.candidates:
        nll = _nll(candidate.text, counts, context_totals, vocabulary)
        details[f"nll:{candidate.doc_id}"] = round(nll, 6)
        out[candidate.doc_id] = clip01((nll - median) / (3.0 * mad) / 2.0)
    return SignalResult(NAME, out, details)


register(compute)


# --------------------------------------------------------------------------- #
# Training and calibration
# --------------------------------------------------------------------------- #


def train(benign_texts: list[str]) -> dict:
    """Build the character n-gram table and the median/MAD of its NLL."""
    counts: Counter = Counter()
    for text in benign_texts:
        padded = _padded(text)
        for index in range(len(padded) - ORDER):
            counts[padded[index : index + ORDER + 1]] += 1

    context_totals: Counter = Counter()
    for gram, value in counts.items():
        context_totals[gram[:-1]] += value

    vocabulary = len({gram[-1] for gram in counts}) or 1
    counts_plain = dict(counts)
    totals_plain = dict(context_totals)

    nlls = [_nll(text, counts_plain, totals_plain, vocabulary) for text in benign_texts]
    ordered = sorted(nlls)
    median = ordered[len(ordered) // 2] if ordered else 0.0
    deviations = sorted(abs(value - median) for value in nlls) or [0.0]
    mad = deviations[len(deviations) // 2] or 1e-6
    return {
        "counts": counts_plain,
        "context_totals": totals_plain,
        "vocabulary": vocabulary,
        "median": median,
        "mad": mad,
        "n_texts": len(benign_texts),
    }
