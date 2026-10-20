"""Secret masking and log scrubbing.

Every byte the pipeline writes to disk (logs, CSVs, reports, state files) passes
through :func:`scrub_secrets` first, and every user-facing message that needs to
refer to a key uses :func:`mask_key`. This is defence in depth: the key should
never reach these code paths at all, but a single accidental ``print(key)`` in a
later work package must not be able to leak it into an artefact.
"""

from __future__ import annotations

import re

#: DeepSeek-style API keys. Deliberately broad: we would rather over-mask.
SECRET_RE = re.compile(r"sk-[A-Za-z0-9_\-]{16,}")

#: Placeholder written when masking a value that is too short to be a real key.
_PLACEHOLDER = "sk-******"


def looks_like_secret(value: object) -> bool:
    """True when ``value`` contains something shaped like an API key."""
    return isinstance(value, str) and bool(SECRET_RE.search(value))


def mask_key(key: str | None) -> str:
    """Return a display-safe representation such as ``sk-******a1b2``.

    The last four characters are kept so that a human can tell two keys apart;
    everything else is replaced. Short or missing values are fully masked.
    """
    if not key:
        return "(missing)"
    if len(key) < 12:
        return "******"
    return f"sk-******{key[-4:]}"


def scrub_secrets(text: str) -> str:
    """Replace every key-shaped token in ``text`` with its masked form."""
    if not text:
        return text
    return SECRET_RE.sub(lambda m: mask_key(m.group(0)), text)


def scrub_mapping(data: dict[str, object]) -> dict[str, object]:
    """Return a copy of ``data`` with key-shaped string values masked."""
    out: dict[str, object] = {}
    for name, value in data.items():
        if isinstance(value, str):
            out[name] = scrub_secrets(value)
        elif isinstance(value, dict):
            out[name] = scrub_mapping(value)  # type: ignore[arg-type]
        else:
            out[name] = value
    return out
