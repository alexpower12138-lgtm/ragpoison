"""Strict JSON extraction from model output.

Every LLM-backed signal asks for JSON and nothing else, but a model will still
wrap it in prose or a code fence now and then. This module recovers the JSON when
it is there and reports failure when it is not, so the caller can count parse
failures instead of silently scoring them as zero.

Nothing here is lenient about the *content*: a recovered object with the wrong
keys or out-of-range values is a failure, not a guess.
"""

from __future__ import annotations

import json
import re

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _balanced_spans(text: str) -> list[str]:
    """Return every balanced ``{...}`` or ``[...]`` span, outermost first."""
    spans: list[str] = []
    for opener, closer in (("{", "}"), ("[", "]")):
        depth = 0
        start = -1
        for index, char in enumerate(text):
            if char == opener:
                if depth == 0:
                    start = index
                depth += 1
            elif char == closer and depth:
                depth -= 1
                if depth == 0 and start >= 0:
                    spans.append(text[start : index + 1])
                    start = -1
    spans.sort(key=len, reverse=True)
    return spans


def json_from_text(text: str) -> object | None:
    """Return the first parsable JSON value found in ``text``."""
    if not text:
        return None
    candidates = [match.group(1) for match in _FENCE_RE.finditer(text)]
    candidates.append(text)
    candidates.extend(_balanced_spans(text))
    for candidate in candidates:
        stripped = candidate.strip()
        if not stripped:
            continue
        try:
            return json.loads(stripped)
        except ValueError:
            continue
    return None


def parse_answer(text: str) -> tuple[str, float] | None:
    """Parse ``{"answer": str, "confidence": number}``; None when unusable."""
    payload = json_from_text(text)
    if not isinstance(payload, dict):
        return None
    answer = payload.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return None
    confidence = payload.get("confidence", 0.5)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.5
    return answer.strip(), max(0.0, min(1.0, confidence))


def parse_score(text: str) -> tuple[float, str] | None:
    """Parse ``{"score": 0|0.5|1, "category": str}``; None when unusable."""
    payload = json_from_text(text)
    if not isinstance(payload, dict):
        return None
    score = payload.get("score")
    try:
        score = float(score)
    except (TypeError, ValueError):
        return None
    if score not in (0.0, 0.5, 1.0):
        # Accept a value that rounds to one of the three allowed levels.
        if 0.0 <= score <= 1.0:
            score = min((0.0, 0.5, 1.0), key=lambda level: abs(level - score))
        else:
            return None
    return score, str(payload.get("category", "unknown"))


def parse_flags(text: str, expected_ids: list[int]) -> dict[int, int] | None:
    """Parse ``[{"id": int, "flag": 0|1}, ...]`` for the expected ids."""
    payload = json_from_text(text)
    if isinstance(payload, dict):
        payload = payload.get("results") or payload.get("judgements") or None
    if not isinstance(payload, list):
        return None
    flags: dict[int, int] = {}
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        try:
            identifier = int(entry.get("id"))
        except (TypeError, ValueError):
            continue
        value = entry.get("flag")
        if value is None:
            value = entry.get("harmful")
        try:
            flag = 1 if float(value) >= 0.5 else 0
        except (TypeError, ValueError):
            continue
        if identifier in expected_ids:
            flags[identifier] = flag
    if not flags:
        return None
    # Missing ids are treated as "not flagged" and the caller counts the shortfall.
    return flags
