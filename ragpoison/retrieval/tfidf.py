"""TF-IDF index used by the cross-consistency signal.

scikit-learn is already a dependency, and its vectoriser is deterministic for a
fixed corpus, so there is no reason to hand-roll this one (unlike BM25, where the
parameterisation had to be explicit).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class TfidfIndex:
    """TF-IDF matrix over a corpus, with cosine similarity lookups."""

    doc_ids: list[str] = field(default_factory=list)
    _matrix = None
    _position: dict[str, int] = field(default_factory=dict)

    def fit(self, documents: list[tuple[str, str]]) -> "TfidfIndex":
        self.doc_ids = [doc_id for doc_id, _text in documents]
        self._position = {doc_id: index for index, doc_id in enumerate(self.doc_ids)}
        vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b",
            strip_accents=None,
            norm="l2",
            smooth_idf=True,
            sublinear_tf=False,
        )
        self._matrix = vectorizer.fit_transform([text for _doc_id, text in documents])
        return self

    @property
    def size(self) -> int:
        return len(self.doc_ids)

    def similarity_matrix(self, doc_ids: list[str]) -> np.ndarray:
        """Pairwise cosine similarity among ``doc_ids``, in the given order."""
        positions = [self._position[doc_id] for doc_id in doc_ids]
        subset = self._matrix[positions]
        return cosine_similarity(subset)

    def mean_similarity_to_others(self, doc_ids: list[str]) -> dict[str, float]:
        """For each id, its mean cosine similarity to the other listed ids."""
        if len(doc_ids) < 2:
            return {doc_id: 0.0 for doc_id in doc_ids}
        matrix = self.similarity_matrix(doc_ids)
        n = matrix.shape[0]
        off_diagonal_sums = matrix.sum(axis=1) - np.diag(matrix)
        means = off_diagonal_sums / (n - 1)
        return {doc_id: float(means[index]) for index, doc_id in enumerate(doc_ids)}
