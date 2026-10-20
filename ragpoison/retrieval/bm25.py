"""Hand-written BM25Okapi.

``rank_bm25`` is not available offline on the target machine and is not needed:
the formula is short, and owning it means the parameters are explicit and the
tie-breaking is deterministic, which the ranking audit in WP2 depends on.

Scoring (Okapi BM25, with the smoothed idf variant that stays positive):

    idf(t)     = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
    score(q, d)= sum_t idf(t) * f(t,d)*(k1+1) / (f(t,d) + k1*(1 - b + b*|d|/avgdl))

Ties are broken by ascending document index so that two runs on the same corpus
produce identical rankings.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from .tokenize import tokenize


@dataclass
class BM25:
    """A small, dependency-free BM25Okapi index."""

    k1: float = 1.5
    b: float = 0.75
    lowercase: bool = True
    strip_punct: bool = True
    drop_stopwords: bool = False

    doc_ids: list[str] = field(default_factory=list)
    _term_freqs: list[Counter] = field(default_factory=list)
    _doc_len: list[int] = field(default_factory=list)
    _idf: dict[str, float] = field(default_factory=dict)
    _avgdl: float = 0.0

    # ---------------------------------------------------------------- build
    def fit(self, documents: list[tuple[str, str]]) -> "BM25":
        """Index ``[(doc_id, text), ...]``."""
        self.doc_ids = [doc_id for doc_id, _ in documents]
        self._term_freqs = []
        self._doc_len = []
        document_freq: Counter = Counter()

        for _doc_id, text in documents:
            tokens = self._tokens(text)
            freqs = Counter(tokens)
            self._term_freqs.append(freqs)
            self._doc_len.append(len(tokens))
            document_freq.update(freqs.keys())

        total = len(documents)
        self._avgdl = (sum(self._doc_len) / total) if total else 0.0
        self._idf = {
            term: math.log(1.0 + (total - df + 0.5) / (df + 0.5))
            for term, df in document_freq.items()
        }
        return self

    def _tokens(self, text: str) -> list[str]:
        return tokenize(
            text,
            lowercase=self.lowercase,
            strip_punct=self.strip_punct,
            drop_stopwords=self.drop_stopwords,
        )

    # ---------------------------------------------------------------- query
    def scores(self, query: str) -> list[float]:
        """Return one score per indexed document, in index order."""
        query_terms = self._tokens(query)
        if not query_terms or not self._term_freqs:
            return [0.0] * len(self._term_freqs)

        avgdl = self._avgdl or 1.0
        norm = self.k1 * (1.0 - self.b + self.b * 1.0)
        cache: dict[str, float] = {}
        for term in set(query_terms):
            idf = self._idf.get(term)
            if idf is None:
                continue
            cache[term] = idf * (self.k1 + 1.0)

        out: list[float] = [0.0] * len(self._term_freqs)
        for index, freqs in enumerate(self._term_freqs):
            doc_len = self._doc_len[index]
            denominator_base = self.k1 * (1.0 - self.b + self.b * doc_len / avgdl)
            total = 0.0
            for term, weight in cache.items():
                freq = freqs.get(term)
                if not freq:
                    continue
                total += weight * freq / (freq + denominator_base)
            out[index] = total
        return out

    def ranked_ids(self, query: str, top_n: int | None = None) -> list[tuple[str, float]]:
        """Return ``[(doc_id, score), ...]`` sorted by score desc, id asc on ties."""
        scored = self.scores(query)
        order = sorted(range(len(scored)), key=lambda i: (-scored[i], i))
        if top_n is not None:
            order = order[:top_n]
        return [(self.doc_ids[i], scored[i]) for i in order]

    def rank_of(self, query: str, doc_id: str) -> int:
        """1-based rank of ``doc_id`` for ``query``; 0 when it is not indexed."""
        if doc_id not in self.doc_ids:
            return 0
        for position, (candidate, _score) in enumerate(self.ranked_ids(query), start=1):
            if candidate == doc_id:
                return position
        return 0

    @property
    def size(self) -> int:
        return len(self.doc_ids)

    @property
    def avgdl(self) -> float:
        return self._avgdl
