"""Fusion: turn a subset of signal columns into one suspicion score.

Three rules, exactly as the specification defines them:

``none``      no defence at all (C0). Every candidate scores 0, so nothing is ever
              filtered and the resulting ASR is the undefended baseline.
``threshold`` ``S_i = mean(active signals)``. Simple, no weights to fit, and the
              only rule whose behaviour can be read straight off the signal means.
``weighted``  ``S_i = w_i * sum_k alpha_k s_k_i`` where ``w_i`` is the retrieval
              prior (S7) and ``alpha`` is fitted on the calibration split. The
              prior multiplies; it is never a verdict of its own, and the formula
              deliberately does not normalise by ``sum alpha`` so that ``alpha``
              changes the scale and therefore interacts with the threshold.
``logistic``  a balanced logistic regression on the calibration split. Reported
              with its coefficients, because a fusion that works only as a black
              box is not a finding.

Everything here is pure: no file access, no model calls. That is what makes the
fifteen ablation rows cost nothing (decision D12).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..signals.base import ABLATION_COLUMNS

#: Fixed starting weights from the specification, used when a fit is unavailable.
DEFAULT_ALPHAS: dict[str, float] = {
    "s1": 0.5,
    "s2": 1.0,
    "s3": 1.5,
    "s4": 1.0,
    "s5": 1.5,
    "s6": 0.5,
}

RULES: tuple[str, ...] = ("none", "threshold", "weighted", "logistic")


@dataclass
class FusionModel:
    """A fitted fusion, able to score one row."""

    signals: list[str]  # ablation names, in matrix order
    rule: str
    threshold: float = 0.0
    alphas: dict[str, float] = field(default_factory=dict)
    coefficients: dict[str, float] = field(default_factory=dict)
    intercept: float = 0.0
    flip_query_sign: bool = False
    fitted: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def columns(self) -> list[str]:
        return [ABLATION_COLUMNS[name][1] for name in self.signals]

    def value(self, row: dict, name: str) -> float:
        column = ABLATION_COLUMNS[name][1]
        raw = float(row.get(column, 0.0))
        if self.flip_query_sign and name == "query":
            return 1.0 - raw
        return raw

    def score(self, row: dict) -> float:
        if self.rule == "none" or not self.signals:
            return 0.0
        if self.rule == "threshold":
            values = [self.value(row, name) for name in self.signals]
            return sum(values) / len(values)
        if self.rule == "weighted":
            total = sum(self.alphas.get(name, 1.0) * self.value(row, name) for name in self.signals)
            prior = float(row.get("s7", 1.0))
            return prior * total
        if self.rule == "logistic":
            linear = self.intercept + sum(
                self.coefficients.get(name, 0.0) * self.value(row, name) for name in self.signals
            )
            return 1.0 / (1.0 + pow(2.718281828459045, -linear))
        raise ValueError(f"unknown fusion rule: {self.rule}")

    def scores(self, rows: list[dict]) -> list[float]:
        return [self.score(row) for row in rows]


def build_model(signals: list[str], rule: str, *, flip_query_sign: bool = False) -> FusionModel:
    if rule not in RULES:
        raise ValueError(f"unknown fusion rule: {rule}")
    model = FusionModel(signals=list(signals), rule=rule, flip_query_sign=flip_query_sign)
    if rule == "weighted":
        model.alphas = {name: DEFAULT_ALPHAS.get(ABLATION_COLUMNS[name][1], 1.0) for name in signals}
    return model
