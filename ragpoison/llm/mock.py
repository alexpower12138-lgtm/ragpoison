"""Deterministic offline responder for the mock transport.

Purpose: exercise the whole pipeline (prompt building, parsing, scoring,
aggregation, cost accounting, report writing) with no network and no key. The
responder answers with the last number that appears in the passage, which mimics
a model that follows its context. It is a plumbing test, not a result: every
artefact written under the mock transport is stamped ``"mock": true``.
"""

from __future__ import annotations

import re

_NUMBER_RE = re.compile(r"\b\d{2,4}\b")


def last_number(prompt: str) -> str:
    found = _NUMBER_RE.findall(prompt)
    return found[-1] if found else "UNKNOWN"
