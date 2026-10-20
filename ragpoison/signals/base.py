"""Signal plumbing: the shared result type, the context, and the registry.

Each signal is a pure function of a :class:`SignalContext` returning a
:class:`SignalResult`. Nothing writes to global state, so signals can be switched,
reordered or re-run from the frozen table without touching anything else.

Scores are always in ``[0, 1]`` with *higher meaning more suspicious*, because the
fusion layer assumes that direction. A signal whose natural direction is the
opposite (a retrieval prior, say) must say so in its docstring and be used as a
weight instead of a verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

from ..config import Config
from ..retrieval.bm25 import BM25
from ..retrieval.tfidf import TfidfIndex

#: Ablation name (as used in ``configs/ablation.yaml``) -> (table, column).
#:
#: The ablation matrix speaks in conceptual names, the tables speak in column
#: names, and the two are not the same. Keeping the mapping here means the fusion
#: layer never has to guess, and the self-test can prove that every name in the
#: matrix resolves to a column that actually exists.
#:
#:   "local" table: results/signals/seed{N}/{family}_inj{n}.jsonl      (WP3, free)
#:   "llm"   table: results/signals/seed{N}/{family}_inj{n}.llm.jsonl  (WP4, paid)
ABLATION_COLUMNS: dict[str, tuple[str, str]] = {
    "query": ("local", "s1"),
    "cross": ("local", "s2"),
    "conflict": ("llm", "s3"),
    "conflict_majority": ("llm", "s3b"),
    "instruction": ("llm", "s4"),
    "multiview": ("llm", "s5"),
    "ppl": ("local", "s6"),
}

#: Column name of the retrieval prior. It is a fusion weight, never a verdict,
#: and it is deliberately absent from the ablation matrix.
PRIOR_COLUMN: tuple[str, str] = ("local", "s7")


def columns_for(names: Iterable[str]) -> list[tuple[str, str]]:
    """Resolve ablation names to ``(table, column)`` pairs, in the given order."""
    missing = [name for name in names if name not in ABLATION_COLUMNS]
    if missing:
        raise KeyError(f"unknown ablation signal name(s): {missing}")
    return [ABLATION_COLUMNS[name] for name in names]


@dataclass(frozen=True)
class Candidate:
    """One retrieved passage, before any judgement."""

    doc_id: str
    text: str
    rank: int
    bm25: float


@dataclass
class SignalContext:
    """Everything a signal may look at. No signal may read anything else."""

    config: Config
    query: str
    candidates: list[Candidate]
    bm25: BM25
    tfidf: TfidfIndex
    calibration: dict
    corpus_size: int
    extra: dict = field(default_factory=dict)

    def ids(self) -> list[str]:
        return [candidate.doc_id for candidate in self.candidates]

    def texts(self) -> dict[str, str]:
        return {candidate.doc_id: candidate.text for candidate in self.candidates}


@dataclass(frozen=True)
class SignalResult:
    """Per-candidate scores plus the raw numbers behind them."""

    name: str
    scores: dict[str, float]
    details: dict[str, float] = field(default_factory=dict)
    enabled: bool = True

    def value(self, doc_id: str) -> float:
        return float(self.scores.get(doc_id, 0.0))


#: A signal is any callable with this shape.
SignalFn = Callable[[SignalContext], SignalResult]

_REGISTRY: dict[str, SignalFn] = {}


def register(fn: SignalFn) -> SignalFn:
    """Decorator used by each signal module."""
    name = fn.__module__.rsplit(".", 1)[-1].split("_", 1)[0]  # s1_query_sim -> s1
    _REGISTRY[name] = fn
    return fn


def registry() -> dict[str, SignalFn]:
    """Import every signal module, then return the populated registry."""
    from . import s1_sim_overfit, s2_cross_consistency, s6_pseudo_ppl, s7_prior  # noqa: F401

    return dict(_REGISTRY)


def get(name: str) -> SignalFn:
    return registry()[name]


def run_signals(context: SignalContext, names: Iterable[str] | None = None) -> dict[str, SignalResult]:
    """Run the requested signals (all of them by default)."""
    available = registry()
    chosen = list(names) if names is not None else sorted(available)
    results: dict[str, SignalResult] = {}
    for name in chosen:
        if name not in available:
            raise KeyError(f"unknown signal: {name}")
        results[name] = available[name](context)
    return results


def clip01(value: float) -> float:
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)
