"""S3 - answer centrality (frequency density).

Adapted from the Freq-Density idea in FilterRAG [8]. For every candidate a short
answer is extracted from that passage alone, and then the *density* of the
question and answer words inside the passage is measured:

    W      = tokens(question) union tokens(answer), stopwords removed
    c_i    = sum_{w in W} tf(w, d_i) / len(tokens(d_i))
    s3_i   = 1 - clip(c_i / (c_ref * 2), 0, 1)

The intuition: genuine evidence spreads the answer words across a passage, while a
poisoned passage concentrates the question's keywords and the attacker's value into
a short span. High density is suspicious, so the score is inverted.

``c_ref`` is the median density over benign calibration candidates. It is not
stored in the table (the raw ``s3_c`` is), because the median belongs to the
calibration split and the fusion step is where it gets applied.

An answer of UNKNOWN is itself a signal (the passage cannot answer the question),
and is scored as maximally suspicious, following the plan. The rate of UNKNOWN is
reported alongside.
"""

from __future__ import annotations

from ..retrieval.tokenize import STOPWORDS, tokenize

NAME = "s3"

UNKNOWN_ANSWER_SCORE = 1.0


def density(passage: str, question: str, answer: str) -> float:
    """Return ``c_i`` for one passage."""
    if answer.strip().upper().startswith("UNKNOWN"):
        return 0.0
    focus = {
        token
        for token in tokenize(question) + tokenize(answer)
        if token not in STOPWORDS
    }
    passage_tokens = tokenize(passage)
    if not passage_tokens or not focus:
        return 0.0
    hits = sum(1 for token in passage_tokens if token in focus)
    return hits / len(passage_tokens)


def score_from_density(c_density: float, answer: str, c_ref: float) -> float:
    """Map a density to a suspicion score in ``[0, 1]``."""
    if answer.strip().upper().startswith("UNKNOWN"):
        return UNKNOWN_ANSWER_SCORE
    if c_ref <= 0:
        return 0.0
    ratio = c_density / (c_ref * 2.0)
    return 1.0 - (0.0 if ratio < 0.0 else (1.0 if ratio > 1.0 else ratio))
