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


def normalize_answer(text: str) -> str:
    """Lowercase, drop punctuation and thousands separators, collapse spaces."""
    lowered = (text or "").lower().replace(",", "")
    cleaned = _STRIP_RE.sub(" ", lowered)
    return _SPACE_RE.sub(" ", cleaned).strip()


def is_unknown(text: str) -> bool:
    return normalize_answer(text).startswith("unknown")


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
