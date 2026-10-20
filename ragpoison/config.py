"""Configuration loading and strict validation.

Design rules (see the project specification, section 3):

* ``configs/default.yaml`` is the single source of defaults.
* Any other configuration is deep-merged on top of the default.
* Unknown keys are an error, not a warning. A typo such as
  ``enable_multiview_judge`` silently not taking effect would invalidate an
  ablation cell, so the loader refuses to start instead.
* A configuration file that contains something shaped like an API key is
  rejected outright (decision D24).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .paths import path as repo_path
from .redact import SECRET_RE

# --------------------------------------------------------------------------- #
# Schema: every allowed key, recursively. Leaves are the expected Python type
# (or a tuple of types). Used both for validation and for documentation.
# --------------------------------------------------------------------------- #

Schema = dict[str, Any]

_SCHEMA: Schema = {
    "runtime": {
        "concurrency": int,
        "budget_cny": (int, float),
        "seed": int,
        "request_timeout_s": (int, float),
        "max_retries": int,
        "log_level": str,
    },
    "llm": {
        "base_url": str,
        "model": str,
        "temperature": (int, float),
        "max_tokens_json": int,
        "max_tokens_answer": int,
        "provider": str,
        "price_input_cny_per_mtok": (int, float),
        "price_input_cached_cny_per_mtok": (int, float),
        "price_output_cny_per_mtok": (int, float),
    },
    "retrieval": {
        "retriever": str,
        "top_n": int,
        "top_k": int,
        "bm25_k1": (int, float),
        "bm25_b": (int, float),
        "lowercase": bool,
        "strip_punct": bool,
        "drop_stopwords": bool,
    },
    "detectors": {
        "enable_query_sim": bool,
        "enable_cross_consistency": bool,
        "enable_answer_conflict": bool,
        "enable_instruction_judge": bool,
        "enable_multiview_judge": bool,
        "enable_pseudo_ppl": bool,
        "multiview": {
            "enable_relevance": bool,
            "enable_instruction": bool,
            "enable_contradiction": bool,
            "enable_inducement": bool,
        },
        "mv_batch_size": int,
        "answer_conflict_topm": int,
    },
    "fusion": {
        "method": str,
        "threshold": (int, float),
        "weights": {
            "query": (int, float),
            "cross": (int, float),
            "conflict": (int, float),
            "instruction": (int, float),
            "multiview": (int, float),
            "ppl": (int, float),
        },
        "prior_weight_scale": (int, float),
    },
    "generation": {
        "prompt_version": str,
        "expect_unknown_token": str,
        "answer_match": str,
    },
    "scan": {
        "enabled": bool,
        "strategy": str,
        "batch_size": int,
        "chunk_max_tokens": int,
        "budget_tokens_per_scan": int,
        "coarse_threshold": (int, float),
        "shard_count": int,
        "schedule_cron": str,
        "scan_version": int,
        "stale_after_days": int,
    },
    "experiment": {
        "attacks": list,
        "injection_counts": list,
        "seeds": list,
        "splits": {"calib": int, "test": int},
        "cache_dir": str,
        "max_instances": int,
        "pilot_queries": int,
        "replicate_queries": int,
    },
}


class ConfigError(ValueError):
    """Raised when a configuration file is missing, malformed or unsafe."""


@dataclass(frozen=True)
class Config:
    """Validated, read-only configuration tree with dotted-path access."""

    data: dict[str, Any]
    source: Path
    overrides: tuple[str, ...] = ()

    def get(self, dotted: str, default: Any = None) -> Any:
        """Return the value at ``dotted`` such as ``retrieval.top_n``."""
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        """Like :meth:`get` but raises when the key is absent."""
        sentinel = object()
        value = self.get(dotted, sentinel)
        if value is sentinel:
            raise ConfigError(f"missing required configuration key: {dotted}")
        return value

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Merge ``overlay`` into ``base`` in place, rejecting unknown keys."""
    for key, value in overlay.items():
        dotted = f"{prefix}{key}"
        if key not in base:
            raise ConfigError(f"unknown configuration key: {dotted}")
        current = base[key]
        if isinstance(value, dict):
            if not isinstance(current, dict):
                raise ConfigError(f"{dotted} must be a scalar, not a mapping")
            _deep_merge(current, value, prefix=f"{dotted}.")
        else:
            _check_type(dotted, value, _schema_at(dotted))
            base[key] = value
    return base


def _schema_at(dotted: str) -> Any:
    node: Any = _SCHEMA
    for part in dotted.split("."):
        node = node[part]
    return node


def _check_type(dotted: str, value: Any, expected: Any) -> None:
    if isinstance(expected, dict):  # pragma: no cover - guarded by caller
        raise ConfigError(f"{dotted} must be a mapping")
    # bool is a subclass of int; reject the confusion explicitly.
    if expected is int and isinstance(value, bool):
        raise ConfigError(f"{dotted} must be an integer, not a boolean")
    if expected is bool and not isinstance(value, bool):
        raise ConfigError(f"{dotted} must be a boolean (true/false)")
    if not isinstance(value, expected):
        names = (
            "/".join(t.__name__ for t in expected)
            if isinstance(expected, tuple)
            else expected.__name__
        )
        raise ConfigError(f"{dotted} must be {names}, got {type(value).__name__}")


def _reject_secrets(raw: str, source: Path) -> None:
    if SECRET_RE.search(raw):
        raise ConfigError(
            f"{source.name} appears to contain an API key. "
            "Keys belong in the secret store (`python run.py secret set`), never in configs."
        )


def load_config(
    name: str | Path = "default.yaml",
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Load ``configs/<name>`` merged over ``configs/default.yaml``.

    ``name`` may be a bare file name inside ``configs/`` or a path to any YAML
    file. ``overrides`` is a nested dict applied last (used by the CLI flags).
    """
    default_path = repo_path("configs", "default.yaml")
    if not default_path.exists():
        raise ConfigError(f"missing default configuration: {default_path}")

    default_raw = default_path.read_text(encoding="utf-8")
    _reject_secrets(default_raw, default_path)
    base = yaml.safe_load(default_raw) or {}
    if not isinstance(base, dict):
        raise ConfigError("configs/default.yaml must contain a mapping at the top level")

    source = default_path
    if str(name) not in {"default.yaml", "default"}:
        candidate = Path(name)
        if not candidate.is_absolute() and not candidate.exists():
            candidate = repo_path("configs", str(name))
        if not candidate.exists():
            raise ConfigError(f"configuration file not found: {name}")
        source = candidate
        raw = candidate.read_text(encoding="utf-8")
        _reject_secrets(raw, candidate)
        overlay = yaml.safe_load(raw) or {}
        if not isinstance(overlay, dict):
            raise ConfigError(f"{candidate.name} must contain a mapping at the top level")
        base = _deep_merge(base, overlay)

    applied: list[str] = []
    if overrides:
        base = _deep_merge(base, overrides)
        applied = [f"override:{k}" for k in overrides]

    _validate_complete(base)
    return Config(data=base, source=source, overrides=tuple(applied))


def _validate_complete(tree: dict[str, Any], schema: Schema | None = None, prefix: str = "") -> None:
    """Ensure no key is missing and no key is extra, recursively."""
    schema = _SCHEMA if schema is None else schema
    for key, expected in schema.items():
        dotted = f"{prefix}{key}"
        if key not in tree:
            raise ConfigError(f"missing configuration key: {dotted}")
        if isinstance(expected, dict):
            if not isinstance(tree[key], dict):
                raise ConfigError(f"{dotted} must be a mapping")
            _validate_complete(tree[key], expected, prefix=f"{dotted}.")
        else:
            _check_type(dotted, tree[key], expected)
    extra = set(tree) - set(schema)
    if extra:
        raise ConfigError(f"unknown configuration key(s): {sorted(extra)}")


def render_config(config: Config) -> str:
    """Return the effective configuration as YAML, with secrets scrubbed."""
    from .redact import scrub_secrets

    body = yaml.safe_dump(config.data, sort_keys=False, allow_unicode=True)
    return scrub_secrets(body)


# --------------------------------------------------------------------------- #
# Ablation matrix
# --------------------------------------------------------------------------- #

#: Signal columns that may appear in the ablation matrix.
ABLATION_SIGNALS: frozenset[str] = frozenset(
    {"query", "cross", "conflict", "conflict_majority", "instruction", "multiview", "ppl"}
)

#: Fusion methods understood by the offline re-fusion step.
FUSION_METHODS: frozenset[str] = frozenset({"none", "threshold", "weighted", "logistic"})

#: Configurations that the specification requires to exist (section 6.6).
REQUIRED_ABLATIONS: frozenset[str] = frozenset(
    {"C0", "C1", "C1f", "C2", "C3", "C3b", "C4", "C5", "C6", "C7", "C8", "C8w", "C8l", "C9", "C10"}
)

_ABLATION_ENTRY_KEYS = {"label", "signals", "fusion", "flip_query_sign"}
_ABLATION_TOP_KEYS = {"version", "all_signals", "ablations"}


@dataclass(frozen=True)
class AblationSpec:
    """One row of the offline ablation matrix."""

    ab_id: str
    label: str
    signals: tuple[str, ...]
    fusion: str
    flip_query_sign: bool = False


def load_ablation_matrix(name: str = "ablation.yaml") -> dict[str, AblationSpec]:
    """Load and validate ``configs/ablation.yaml``.

    This file is a data table, not an experiment configuration, so it is
    validated by its own rules and is *not* merged onto ``default.yaml``.
    """
    candidate = Path(name)
    if not candidate.is_absolute() and not candidate.exists():
        candidate = repo_path("configs", str(name))
    if not candidate.exists():
        raise ConfigError(f"ablation matrix not found: {name}")

    raw = candidate.read_text(encoding="utf-8")
    _reject_secrets(raw, candidate)
    payload = yaml.safe_load(raw) or {}
    if not isinstance(payload, dict):
        raise ConfigError("ablation matrix must contain a mapping at the top level")

    unknown = set(payload) - _ABLATION_TOP_KEYS
    if unknown:
        raise ConfigError(f"unknown key(s) in the ablation matrix: {sorted(unknown)}")

    declared = payload.get("all_signals") or []
    if not isinstance(declared, list) or set(declared) != ABLATION_SIGNALS:
        raise ConfigError(
            "all_signals must list exactly: " + ", ".join(sorted(ABLATION_SIGNALS))
        )

    ablations = payload.get("ablations")
    if not isinstance(ablations, dict) or not ablations:
        raise ConfigError("ablation matrix has no ablations")

    specs: dict[str, AblationSpec] = {}
    for ab_id, entry in ablations.items():
        if not isinstance(entry, dict):
            raise ConfigError(f"{ab_id}: each ablation must be a mapping")
        extra = set(entry) - _ABLATION_ENTRY_KEYS
        if extra:
            raise ConfigError(f"{ab_id}: unknown key(s) {sorted(extra)}")
        signals = entry.get("signals", [])
        if not isinstance(signals, list):
            raise ConfigError(f"{ab_id}: signals must be a list")
        bad = set(signals) - ABLATION_SIGNALS
        if bad:
            raise ConfigError(f"{ab_id}: unknown signal(s) {sorted(bad)}")
        fusion = entry.get("fusion", "threshold")
        if fusion not in FUSION_METHODS:
            raise ConfigError(f"{ab_id}: fusion must be one of {sorted(FUSION_METHODS)}")
        flip = bool(entry.get("flip_query_sign", False))
        if fusion == "none" and signals:
            raise ConfigError(f"{ab_id}: fusion 'none' must not enable any signal")
        specs[ab_id] = AblationSpec(
            ab_id=ab_id,
            label=str(entry.get("label", ab_id)),
            signals=tuple(signals),
            fusion=str(fusion),
            flip_query_sign=flip,
        )

    missing = REQUIRED_ABLATIONS - set(specs)
    if missing:
        raise ConfigError(f"ablation matrix is missing required configurations: {sorted(missing)}")
    return specs
