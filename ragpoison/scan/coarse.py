"""Stage-1 of the corpus audit: a free, per-chunk suspicion score.

There is no query when auditing a whole corpus, so the query-time signals are not
available. The specification defines the coarse score as a weighted sum of five
components that cost nothing:

    coarse = 0.35 * Rule + 0.20 * LenZ + 0.20 * PPLz + 0.15 * TfidfOutlier + 0.10 * Dup

* ``Rule`` looks for the surface traces an attacker leaves: meta-discourse about
  sources, instructions addressed to a reader, a verbatim question, meta-commentary.
* ``LenZ`` is how unusual the passage length is, in robust deviations from the median.
* ``PPLz`` is the pseudo-perplexity from S6, reusing the character trigram model that
  is already calibrated on benign passages.
* ``TfidfOutlier`` is one minus the cosine between the passage and the corpus centroid:
  a passage that does not look like the rest of the corpus is worth a look.
* ``Dup`` is the Jaccard similarity of the passage to the most similar *other* passage.
  The specification calls it duplication and that is what it means: a passage that
  repeats another one verbatim, or repeats a template, is suspicious. Computing all
  2.2 million pairs is wasteful, so candidates are found through the rarest shingles of
  each passage - the pairs that could possibly be similar - and only those are compared
  exactly. A first attempt used corpus shingle frequency instead, which saturates at
  one for every passage in a 2100-passage corpus and therefore measured nothing.

Only chunks above ``tau_c`` (0.4 by default) reach the paid stage.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from ..signals.s6_pseudo_ppl import ORDER, _nll, _padded

#: Surface traces of a manipulated entry. Kept deliberately mechanical: this stage is
#: allowed to be crude because it only decides what is worth paying for.
RULE_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"\b(ignore|disregard|forget)\b", re.IGNORECASE),
    re.compile(r"\b(do not|don't|never)\s+(trust|believe|use)\b", re.IGNORECASE),
    re.compile(r"\b(earlier|previous|older)\s+(references?|sources?|records?|data)\b", re.IGNORECASE),
    re.compile(r"\b(outdated|superseded|deprecated|amended|corrected)\b", re.IGNORECASE),
    re.compile(r"\b(latest|more recent|newer)\s+(data|records?|sources?|figures?)\b", re.IGNORECASE),
    re.compile(r"\b(answer|reply|respond)\s+(with|only|exactly|that)\b", re.IGNORECASE),
    re.compile(r"\bnote to (the )?(reader|model|assistant)\b", re.IGNORECASE),
    re.compile(r"\?\s*$", re.MULTILINE),
)

#: Weight of each component, exactly as the specification fixes them.
WEIGHTS: dict[str, float] = {
    "rule": 0.35,
    "lenz": 0.20,
    "pplz": 0.20,
    "outlier": 0.15,
    "dup": 0.10,
}


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[len(ordered) // 2])


def _mad(values: list[float], centre: float) -> float:
    deviations = sorted(abs(value - centre) for value in values)
    if not deviations:
        return 1.0
    return float(deviations[len(deviations) // 2]) or 1.0


def _clip(value: float) -> float:
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)


def rule_score(text: str) -> float:
    hits = sum(len(pattern.findall(text)) for pattern in RULE_PATTERNS)
    return min(1.0, hits / 2.0)


def shingles(text: str, size: int = 3) -> set[int]:
    lowered = text.lower()
    return {hash(lowered[index : index + size]) for index in range(len(lowered) - size + 1)}


def coarse_scores(
    texts: dict[str, str],
    *,
    ppl_statistics: dict | None = None,
    dup_min_frequency: int = 5,
) -> dict[str, dict[str, float]]:
    """Score every chunk with the five components and their weighted sum."""
    ids = sorted(texts)
    if not ids:
        return {}

    lengths = [len(texts[chunk].split()) for chunk in ids]
    length_centre = _median([float(value) for value in lengths])
    length_mad = _mad([float(value) for value in lengths], length_centre)

    # Bag-of-words cosine against the corpus centroid, done by hand: the vocabulary is
    # a few thousand words, so this is cheap and adds no dependency.
    tokens = {chunk: texts[chunk].lower().split() for chunk in ids}
    document_frequency: Counter = Counter()
    for chunk in ids:
        document_frequency.update(set(tokens[chunk]))
    vocabulary = len(document_frequency) or 1
    weights = {
        word: math.log((1 + len(ids)) / (1 + count)) + 1.0
        for word, count in document_frequency.items()
    }
    vectors = {
        chunk: Counter({word: count * weights[word] for word, count in Counter(tokens[chunk]).items()})
        for chunk in ids
    }
    centroid: Counter = Counter()
    for vector in vectors.values():
        centroid.update(vector)
    centroid_norm = math.sqrt(sum(value * value for value in centroid.values())) or 1.0

    shingle_sets = {chunk: shingles(texts[chunk]) for chunk in ids}
    postings: dict[int, list[int]] = {}
    for index, chunk in enumerate(ids):
        for shingle in shingle_sets[chunk]:
            postings.setdefault(shingle, []).append(index)

    if ppl_statistics:
        ppl_counts = ppl_statistics.get("counts") or {}
        ppl_context = ppl_statistics.get("context_totals") or {}
        ppl_vocabulary = int(ppl_statistics.get("vocabulary") or 1)
        ppl_median = float(ppl_statistics.get("median") or 0.0)
        ppl_mad = float(ppl_statistics.get("mad") or 1.0)
        raw = {
            chunk: _nll(texts[chunk], ppl_counts, ppl_context, ppl_vocabulary) for chunk in ids
        }
        scales = [_clip((raw[chunk] - ppl_median) / (3.0 * ppl_mad) / 2.0) for chunk in ids]
    else:
        scales = [0.0 for _ in ids]

    outlier_raw: dict[str, float] = {}
    for chunk in ids:
        vector = vectors[chunk]
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        dot = sum(value * centroid.get(word, 0) for word, value in vector.items())
        cosine = dot / (norm * centroid_norm)
        outlier_raw[chunk] = 1.0 - cosine
    outlier_max = max(outlier_raw.values()) or 1.0

    out: dict[str, dict[str, float]] = {}
    for position, chunk in enumerate(ids):
        shingle_set = shingle_sets[chunk]
        duplicate = 0.0
        if shingle_set:
            # Candidate neighbours: passages sharing the rarest shingles of this one.
            # A passage whose rare shingles nobody else has cannot be a duplicate.
            rarest = sorted(shingle_set, key=lambda shingle: len(postings[shingle]))[:5]
            candidates: set[int] = set()
            for shingle in rarest:
                candidates.update(postings[shingle])
            candidates.discard(position)
            for other in list(candidates)[:200]:
                shared = len(shingle_set & shingle_sets[ids[other]])
                if not shared:
                    continue
                union = len(shingle_set) + len(shingle_sets[ids[other]]) - shared
                duplicate = max(duplicate, shared / union if union else 0.0)
        length_z = abs(len(tokens[chunk]) - length_centre) / (3.0 * length_mad)
        components = {
            "rule": rule_score(texts[chunk]),
            "lenz": _clip(length_z),
            "pplz": scales[position],
            "outlier": _clip(outlier_raw[chunk] / outlier_max),
            "dup": _clip(duplicate),
        }
        total_score = sum(WEIGHTS[name] * value for name, value in components.items())
        out[chunk] = {**{name: round(value, 6) for name, value in components.items()},
                      "coarse": round(total_score, 6)}
    return out
