"""Loading a generated bundle back from disk.

The benchmark is regenerable in seconds, so it is never committed; every work
package that needs it reads it through this module and verifies the SHA-256
manifest first, so a stale or hand-edited file cannot silently change a result.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from ..paths import path as repo_path

CORPUS_FIELDS = ("corpus_id", "attack_family", "injection")


class BundleError(RuntimeError):
    """Raised when the generated data is missing or does not match its manifest."""


@dataclass
class Bundle:
    seed: int
    root: Path
    facts: list[dict] = field(default_factory=list)
    clean: list[dict] = field(default_factory=list)
    queries: list[dict] = field(default_factory=list)
    distractors: list[dict] = field(default_factory=list)
    poisoned: list[dict] = field(default_factory=list)
    splits: dict = field(default_factory=dict)
    corpora: dict[str, dict] = field(default_factory=dict)
    manifest: dict = field(default_factory=dict)

    def texts(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for row in self.clean:
            mapping[row["passage_id"]] = row["text"]
        for row in self.distractors:
            mapping[row["passage_id"]] = row["text"]
        for row in self.poisoned:
            mapping[row["passage_id"]] = row["text"]
        return mapping

    def query_by_id(self) -> dict[str, dict]:
        return {row["query_id"]: row for row in self.queries}

    def clean_id_for_fact(self, fact_id: str) -> str:
        for row in self.clean:
            if row["fact_id"] == fact_id:
                return row["passage_id"]
        raise BundleError(f"no clean passage for fact {fact_id}")


def bundle_root(seed: int, root: Path | None = None) -> Path:
    if root is not None:
        return root
    return repo_path("data", "synthetic", f"seed{seed}")


def _read_json(target: Path) -> object:
    if not target.exists():
        raise BundleError(
            f"missing {target}. Generate the benchmark first:  python run.py gen-data --seed {target.parent.name[-2:]}"
        )
    return json.loads(target.read_text(encoding="utf-8"))


def verify_manifest(root: Path) -> tuple[bool, list[str]]:
    """Check every file listed in ``manifest.json`` against its SHA-256."""
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return False, ["manifest.json is missing"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded: dict[str, str] = manifest.get("sha256", {})
    problems: list[str] = []
    for name, digest in sorted(recorded.items()):
        candidate = root / name
        if not candidate.exists():
            problems.append(f"{name}: missing")
            continue
        actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if actual != digest:
            problems.append(f"{name}: sha256 mismatch")
    return (not problems), problems


def load_bundle(seed: int, root: Path | None = None, verify: bool = True) -> Bundle:
    """Load and (by default) integrity-check the bundle for ``seed``."""
    base = bundle_root(seed, root)
    if not base.exists():
        raise BundleError(f"{base} does not exist; run:  python run.py gen-data --seed {seed}")

    if verify:
        ok, problems = verify_manifest(base)
        if not ok:
            raise BundleError(
                "generated data does not match its manifest: "
                + "; ".join(problems[:5])
                + ". Regenerate with:  python run.py gen-data --seed "
                + str(seed)
            )

    bundle = Bundle(
        seed=seed,
        root=base,
        facts=_read_json(base / "facts.json"),  # type: ignore[arg-type]
        clean=_read_json(base / "clean_passages.json"),  # type: ignore[arg-type]
        queries=_read_json(base / "queries.json"),  # type: ignore[arg-type]
        distractors=_read_json(base / "distractor_passages.json"),  # type: ignore[arg-type]
        poisoned=_read_json(base / "poisoned_passages.json"),  # type: ignore[arg-type]
        splits=_read_json(base / "splits.json"),  # type: ignore[arg-type]
        manifest=json.loads((base / "manifest.json").read_text(encoding="utf-8")),
    )
    corpora: dict[str, dict] = {}
    for candidate in sorted((base / "corpora").glob("*.json")):
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        corpora[payload["corpus_id"]] = payload
    if not corpora:
        raise BundleError(f"no corpus manifests under {base / 'corpora'}")
    bundle.corpora = corpora
    return bundle
