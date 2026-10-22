"""Deterministic offline responders for the mock transport.

Purpose: exercise the whole pipeline (prompt building, JSON parsing, scoring,
aggregation, cost accounting, report writing) with no network and no key. Every
artefact written under the mock transport is stamped ``"mock": true``.

Two responders:

``last_number``
    Answers with the last number that appears in the prompt. Mimics a model that
    follows its context. Used by the generation stages.
``signal_responder``
    Returns well-formed JSON in the shape each signal prompt asks for, so the
    parsers and the aggregation get exercised instead of always failing.
"""

from __future__ import annotations

import re

# NOTE: the responder is not part of the cache key, so after changing a response shape
# the mock cache has to be deleted (cache/llm_mock) or the old answers will be served.
_NUMBER_RE = re.compile(r"\b\d{2,4}\b")
_PASSAGE_ID_RE = re.compile(r"^\[(\d+)\]", re.MULTILINE)


def last_number(prompt: str) -> str:
    found = _NUMBER_RE.findall(prompt)
    return found[-1] if found else "UNKNOWN"


def signal_responder(prompt: str) -> str:
    """Return JSON matching whichever signal prompt was sent."""
    if '"answer"' in prompt:
        answer = last_number(prompt)
        return '{"answer": "%s", "confidence": 0.6}' % answer
    if '"score"' in prompt:
        return '{"score": 0.5, "category": "meta_discourse", "reason": "mock trace"}'
    if '"harmful"' in prompt:
        ids = [int(match.group(1)) for match in _PASSAGE_ID_RE.finditer(prompt)]
        if not ids:
            ids = [1]
        entries = ", ".join(
            '{"id": %d, "harmful": 0, "category": "genuine", "reason": "mock verdict"}' % identifier
            for identifier in ids
        )
        return f"[{entries}]"
    if '"flag"' in prompt:
        ids = [int(match.group(1)) for match in _PASSAGE_ID_RE.finditer(prompt)]
        if not ids:
            ids = [1]
        entries = ", ".join(
            '{"id": %d, "flag": 0, "reason": "mock judgement"}' % identifier
            for identifier in ids
        )
        return f"[{entries}]"
    return "UNKNOWN"

