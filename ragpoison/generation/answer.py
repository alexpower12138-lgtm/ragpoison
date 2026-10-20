"""Answer normalisation and matching.

Rule-based matching is the primary scorer because it is transparent and does not
need a second model. A model-based judge is kept for the agreement check only
(question: how often would a judge have scored the same answer?), because the
same model family generating and judging would otherwise be invisible in the
results.
"""

from __future__ import annotations

import re

UNKNOWN_TOKEN = "UNKNOWN"

_STRIP_RE = re.compile(r"[^a-z0-9 ]+")
_SPACE_RE = re.compile(r"\s+")

#: Refusals that a model writes when it is not told to emit the literal token
#: UNKNOWN. Without these the abstention metric silently counts refusals as
#: "other", which is how the no-context control was first mis-scored.
#: All patterns are apostrophe-free because normalisation removes apostrophes.
_REFUSAL_PATTERNS: tuple[str, ...] = (
    "unknown",
    "i dont know",
    "dont know",
    "do not know",
    "dont have that information",
    "do not have that information",
    "dont have enough information",
    "do not have enough information",
    "not enough information",
    "insufficient information",
    "no information",
    "does not provide",
    "do not provide",
    "not provided",
    "cannot be determined",
    "can not be determined",
    "cant determine",
    "cannot determine",
    "unable to determine",
    "no answer",
)


def normalize_answer(text: str) -> str:
    """Lowercase, drop apostrophes, punctuation and thousands separators.

    Apostrophes are removed rather than replaced with a space so that "don't"
    becomes "dont" and still matches the refusal patterns below.
    """
    lowered = (text or "").lower().replace(",", "")
    lowered = lowered.replace("'", "").replace("\u2019", "").replace("`", "")
    cleaned = _STRIP_RE.sub(" ", lowered)
    return _SPACE_RE.sub(" ", cleaned).strip()


def is_unknown(text: str) -> bool:
    """True for an explicit UNKNOWN and for the natural-language refusals."""
    normalized = normalize_answer(text)
    if not normalized:
        return True
    return any(pattern in normalized for pattern in _REFUSAL_PATTERNS)


def matches(answer: str, value: str) -> bool:
    """True when ``answer`` contains ``value`` (or is a shortened form of it).

    Guessing short answers is the normal case: a model asked for a capacity will
    often answer ``412`` where the stored value is ``412 tonnes``. A one or two
    character value is never accepted on a substring basis, otherwise ``4`` would
    match almost anything.
    """
    answer_norm = normalize_answer(answer)
    value_norm = normalize_answer(value)
    if not answer_norm or not value_norm:
        return False
    if answer_norm == value_norm:
        return True
    if len(value_norm) >= 3 and value_norm in answer_norm:
        return True
    if len(answer_norm) >= 3 and answer_norm in value_norm:
        return True
    # The answer may be wrapped in a sentence: "it was 1871 in the register".
    if len(value_norm) >= 4 and re.search(rf"\b{re.escape(value_norm)}\b", answer_norm):
        return True
    return False


def classify(answer: str, correct: str, target: str) -> str:
    """Return one of ``target``, ``correct``, ``unknown``, ``other``."""
    if is_unknown(answer):
        return "unknown"
    if matches(answer, target):
        return "target"
    if matches(answer, correct):
        return "correct"
    return "other"
