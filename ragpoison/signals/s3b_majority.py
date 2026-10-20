"""S3b - naive majority-vote conflict. The negative control.

Every candidate is asked the question on its own; the answers are then clustered
and a candidate is suspicious when its answer is in the minority:

    support_i = share of the pool whose normalised answer equals a_i
    s3b_i     = 1 - support_i

This was the original S3 design. It is kept because it fails in an instructive
way, and the report quantifies the failure: in a pool of fifteen candidates where
fourteen are about other entities, every candidate has a different answer, so the
support of *every* candidate is about 1/15 and the signal degenerates into a
near-constant. The plan predicted this; ablation cell C3b measures it.
"""

from __future__ import annotations

from collections import Counter

from ..generation.answer import normalize_answer

NAME = "s3b"


def scores(answers: dict[str, str]) -> dict[str, float]:
    """Return ``1 - support`` for every candidate in the pool."""
    if not answers:
        return {}
    normalised = {doc_id: normalize_answer(answer) for doc_id, answer in answers.items()}
    counts = Counter(normalised.values())
    total = sum(counts.values()) or 1
    return {doc_id: 1.0 - counts[key] / total for doc_id, key in normalised.items()}


def support(answers: dict[str, str]) -> dict[str, float]:
    """Return the raw support share, for reporting."""
    if not answers:
        return {}
    normalised = {doc_id: normalize_answer(answer) for doc_id, answer in answers.items()}
    counts = Counter(normalised.values())
    total = sum(counts.values()) or 1
    return {doc_id: counts[key] / total for doc_id, key in normalised.items()}
