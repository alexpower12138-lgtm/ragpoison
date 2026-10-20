"""S3 - answer centrality (frequency density).

Adapted from the Freq-Density idea in FilterRAG [8]. For every candidate a short
answer is extracted from that passage alone, and then the *density* of the
question and answer words inside the passage is measured:

    W      = tokens(question) [+ tokens(answer) when an answer was found], no stopwords
    c_i    = sum_{w in W} tf(w, d_i) / len(tokens(d_i))
    s3_i   = clip(c_i / (2 * c_ref), 0, 1)

High concentration is suspicious, so the score rises with density.

Note on the sign: the plan wrote ``s3 = 1 - clip(c / (2 c_ref))`` while stating the
opposite rationale ("a poisoned passage compresses the question and answer words
into a short span"). Those two cannot both be right, and the formula was the wrong
one - the same class of mistake as the sign error in the first version of S1. The
implementation follows the rationale.

**This signal is expected to fail on this benchmark, and measuring that failure is
the point of ablation cell C3.** Two structural reasons, both verified on real
model output:

1. Every chunk here is a short, focused knowledge-base entry of 80-150 words. In a
   corpus of uniformly concentrated chunks there is nothing for a concentration
   measure to separate.
2. In a query-time pool exactly two candidates can answer the question - the
   poisoned passage and the genuine evidence - and both of them concentrate the
   question's words. Measured densities were 0.086 (poison) against 0.097
   (genuine), i.e. no separation at all.

The first version of this signal also scored an UNKNOWN extraction as maximally
suspicious. That was worse than useless: 87% of candidates cannot answer a question
about a different entity, so the signal saturated and ended up flagging the *clean*
passages hardest. UNKNOWN is now simply "no answer available": the density is then
computed from the question tokens alone, and the event is recorded separately as
``s3_unknown`` instead of being turned into a score.
"""

from __future__ import annotations

from ..retrieval.tokenize import STOPWORDS, tokenize

NAME = "s3"


def density(passage: str, question: str, answer: str) -> float:
    """Return ``c_i`` for one passage."""
    focus = {token for token in tokenize(question) if token not in STOPWORDS}
    if not answer.strip().upper().startswith("UNKNOWN"):
        focus |= {token for token in tokenize(answer) if token not in STOPWORDS}
    passage_tokens = tokenize(passage)
    if not passage_tokens or not focus:
        return 0.0
    hits = sum(1 for token in passage_tokens if token in focus)
    return hits / len(passage_tokens)


def score_from_density(c_density: float, c_ref: float) -> float:
    """Map a density to a suspicion score in ``[0, 1]`` (denser is more suspicious)."""
    if c_ref <= 0:
        return 0.0
    ratio = c_density / (2.0 * c_ref)
    return 0.0 if ratio < 0.0 else (1.0 if ratio > 1.0 else ratio)
