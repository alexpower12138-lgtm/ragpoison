"""LLM plumbing: secret resolution (this package, WP0) and the API client,
cache, prompt registry and response parsing (WP4).
"""

from __future__ import annotations

from .secretstore import clear_stored, read_key, set_key, status, verify_key

__all__ = ["clear_stored", "read_key", "set_key", "status", "verify_key"]
