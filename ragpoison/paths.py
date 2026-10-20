"""Filesystem layout helpers.

The repository root is the directory that contains ``run.py``; everything the
pipeline writes at runtime lives below it. Generated directories are created on
demand because git does not track empty directories.
"""

from __future__ import annotations

from pathlib import Path

# ragpoison/ragpoison/paths.py -> ragpoison/ragpoison -> ragpoison
REPO_ROOT: Path = Path(__file__).resolve().parents[1]

#: Directories created on demand by :func:`ensure_dirs`.
GENERATED_DIRS: tuple[str, ...] = (
    "data",
    "cache",
    "logs",
    "results",
    "figures",
    "reports",
    "prompts_snapshot",
    ".secrets",
)


def path(*parts: str) -> Path:
    """Return an absolute path inside the repository root."""
    return REPO_ROOT.joinpath(*parts)


def ensure_dirs(extra: tuple[str, ...] = ()) -> list[Path]:
    """Create every generated directory and return the ones that now exist."""
    created: list[Path] = []
    for name in GENERATED_DIRS + extra:
        target = path(name)
        target.mkdir(parents=True, exist_ok=True)
        created.append(target)
    return created


def rel(target: Path) -> str:
    """Render a path relative to the repository root, for logs and reports."""
    try:
        return target.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(target)
