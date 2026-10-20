"""Model client: caching, retries, concurrency, cost accounting.

Three properties matter more than speed here:

1. **The key is never handled by this module.** It is read through
   :mod:`ragpoison.llm.secretstore`, which removes it from ``os.environ``.
2. **Every call is cached on disk** and every call is logged, so an interrupted
   run resumes without paying twice.
3. **A hard budget cap** stops the run before it can overspend. The cap is in CNY
   because that is what the provider bills in.

A ``mock`` transport is built in so the whole pipeline (parsing, scoring,
reporting, accounting) can be exercised with no network and no key. It is used by
the offline self-tests and must never be used for reported results.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from ..config import Config
from ..paths import path as repo_path
from ..redact import scrub_secrets
from . import secretstore

#: Prices used only for the budget guard, in CNY per million tokens. They are
#: configuration values because prices change; the numbers reported in the paper
#: come from the provider's own usage counters, not from this table.
DEFAULT_PRICE_INPUT = 2.0
DEFAULT_PRICE_INPUT_CACHED = 0.5
DEFAULT_PRICE_OUTPUT = 8.0


class BudgetExceeded(RuntimeError):
    """Raised when the accumulated spend would exceed the configured cap."""


class LLMError(RuntimeError):
    """Raised when a call fails after every retry."""


@dataclass(frozen=True)
class Call:
    """One request. ``tag`` carries the provenance back into the report."""

    system: str
    user: str
    stage: str
    max_tokens: int
    extra: dict = field(default_factory=dict)


@dataclass
class Result:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    cache_hit: bool
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def estimate_tokens(text: str) -> int:
    """Character-based estimate. Deliberately conservative (rounds up)."""
    return max(1, math.ceil(len(text) / 4))


class LLMClient:
    """Thread-safe client with an on-disk cache and a spend ledger."""

    def __init__(
        self,
        config: Config,
        *,
        concurrency: int | None = None,
        budget_cny: float | None = None,
        cache_dir: Path | None = None,
        mock: bool = False,
        mock_responder: Callable[[str], str] | None = None,
        quiet: bool = False,
    ) -> None:
        self.config = config
        self.model = str(config.get("llm.model"))
        self.base_url = str(config.get("llm.base_url")).rstrip("/")
        self.temperature = float(config.get("llm.temperature"))
        self.timeout_s = float(config.get("runtime.request_timeout_s"))
        self.max_retries = int(config.get("runtime.max_retries"))
        self.concurrency = int(concurrency or config.get("runtime.concurrency"))
        self.budget_cny = float(
            budget_cny if budget_cny is not None else config.get("runtime.budget_cny")
        )
        self.price_input = float(config.get("llm.price_input_cny_per_mtok", DEFAULT_PRICE_INPUT))
        self.price_input_cached = float(
            config.get("llm.price_input_cached_cny_per_mtok", DEFAULT_PRICE_INPUT_CACHED)
        )
        self.price_output = float(config.get("llm.price_output_cny_per_mtok", DEFAULT_PRICE_OUTPUT))
        self.mock = mock
        self.mock_responder = mock_responder or (lambda prompt: "UNKNOWN")
        self.quiet = quiet

        self.cache_dir = cache_dir or repo_path(
            "cache", "llm_mock" if mock else "llm"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Mock and real traffic are logged to different files: a mock run must
        # never be mistaken for paid work in the spend record.
        self.log_path = repo_path("logs", "llm_calls_mock.jsonl" if mock else "llm_calls.jsonl")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._key: str | None = None
        self.calls = 0
        self.cache_hits = 0
        self.failures = 0
        self.spent_cny = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.per_stage: dict[str, dict[str, float]] = {}

    # ---------------------------------------------------------------- helpers
    def _api_key(self) -> str:
        if self.mock:
            return "mock-key"
        with self._lock:
            if self._key is None:
                key, _source = secretstore.read_key(required=True)
                self._key = key
            return self._key or ""

    def _cache_key(self, call: Call) -> str:
        payload = {
            # The transport is part of the key: a mock answer must never be served
            # to a real run, and the two live in separate directories as well.
            "transport": "mock" if self.mock else "http",
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": call.max_tokens,
            "system": call.system,
            "user": call.user,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return digest

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _charge(self, stage: str, cost: float, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            if self.spent_cny + cost > self.budget_cny:
                raise BudgetExceeded(
                    f"budget cap reached: spent {self.spent_cny:.2f} CNY of {self.budget_cny:.2f} CNY "
                    f"before stage {stage}. Raise runtime.budget_cny or stop here; the cache keeps "
                    "everything already paid for."
                )
            self.spent_cny += cost
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            bucket = self.per_stage.setdefault(stage, {"calls": 0, "cost_cny": 0.0})
            bucket["calls"] += 1
            bucket["cost_cny"] += cost

    def _log(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with self.log_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(scrub_secrets(line) + "\n")

    def _cost(self, prompt_tokens: int, completion_tokens: int, cached_input: int = 0) -> float:
        fresh = max(prompt_tokens - cached_input, 0)
        return (
            fresh / 1e6 * self.price_input
            + cached_input / 1e6 * self.price_input_cached
            + completion_tokens / 1e6 * self.price_output
        )

    # ---------------------------------------------------------------- transport
    def _post(self, payload: dict) -> dict:
        if self.mock:
            user = payload["messages"][-1]["content"]
            text = self.mock_responder(user)
            return {
                "choices": [{"message": {"content": text}}],
                "usage": {
                    "prompt_tokens": estimate_tokens(json.dumps(payload["messages"])),
                    "completion_tokens": estimate_tokens(text),
                },
            }

        import requests

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key()}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout_s,
        )
        if response.status_code == 200:
            return response.json()
        raise LLMError(f"HTTP {response.status_code}: {response.text[:200]}")

    def _call_once(self, call: Call) -> Result:
        key = self._cache_key(call)
        cache_file = self._cache_path(key)
        if cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                with self._lock:
                    self.cache_hits += 1
                self._log(
                    {
                        "stage": call.stage,
                        "cache_hit": True,
                        "model": self.model,
                        "tag": call.extra,
                        "prompt_tokens": cached.get("prompt_tokens", 0),
                        "completion_tokens": cached.get("completion_tokens", 0),
                    }
                )
                return Result(
                    text=cached["text"],
                    prompt_tokens=int(cached.get("prompt_tokens", 0)),
                    completion_tokens=int(cached.get("completion_tokens", 0)),
                    latency_ms=0,
                    cache_hit=True,
                )
            except (OSError, ValueError, KeyError):
                cache_file.unlink(missing_ok=True)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": call.system},
                {"role": "user", "content": call.user},
            ],
            "temperature": self.temperature,
            "max_tokens": call.max_tokens,
            "stream": False,
        }

        last_error = "unknown error"
        for attempt in range(1, self.max_retries + 1):
            started = time.perf_counter()
            try:
                raw = self._post(payload)
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                text = (raw.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
                usage = raw.get("usage") or {}
                prompt_tokens = int(
                    usage.get("prompt_tokens") or estimate_tokens(json.dumps(payload["messages"]))
                )
                completion_tokens = int(
                    usage.get("completion_tokens") or estimate_tokens(text)
                )
                cached_input = int(usage.get("prompt_cache_hit_tokens") or 0)
                cost = 0.0 if self.mock else self._cost(prompt_tokens, completion_tokens, cached_input)
                self._charge(call.stage, cost, prompt_tokens, completion_tokens)

                cache_file.write_text(
                    json.dumps(
                        {
                            "text": text,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "model": self.model,
                        },
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
                with self._lock:
                    self.calls += 1
                self._log(
                    {
                        "stage": call.stage,
                        "cache_hit": False,
                        "model": self.model,
                        "tag": call.extra,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "cached_input_tokens": cached_input,
                        "latency_ms": elapsed_ms,
                        "cost_cny": round(cost, 6),
                    }
                )
                return Result(text, prompt_tokens, completion_tokens, elapsed_ms, cache_hit=False)

            except BudgetExceeded:
                raise
            except Exception as exc:  # noqa: BLE001 - classified below
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt >= self.max_retries:
                    break
                # Exponential backoff with jitter; a rate-limit needs a longer pause.
                delay = min(2 ** (attempt - 1) * 1.5, 20.0) + random.uniform(0, 0.5)
                if "429" in last_error:
                    delay *= 2
                time.sleep(delay)

        with self._lock:
            self.failures += 1
        self._log({"stage": call.stage, "ok": False, "error": scrub_secrets(last_error)})
        return Result("", 0, 0, 0, cache_hit=False, error=scrub_secrets(last_error))

    # ---------------------------------------------------------------- public
    def complete(self, call: Call) -> Result:
        return self._call_once(call)

    def complete_many(
        self, calls: Iterable[Call], *, progress_every: int = 100, label: str = ""
    ) -> list[Result]:
        """Run ``calls`` with a bounded thread pool, preserving input order."""
        ordered = list(calls)
        results: list[Result | None] = [None] * len(ordered)
        if not ordered:
            return []
        total = len(ordered)
        done = 0
        with ThreadPoolExecutor(max_workers=max(1, self.concurrency)) as pool:
            futures = {pool.submit(self._call_once, call): index for index, call in enumerate(ordered)}
            try:
                for future in as_completed(futures):
                    index = futures[future]
                    results[index] = future.result()
                    done += 1
                    if not self.quiet and progress_every and done % progress_every == 0:
                        print(
                            f"  {label}{done}/{total}  spent {self.spent_cny:.2f} CNY "
                            f"cache {self.cache_hits}/{self.calls + self.cache_hits}",
                            flush=True,
                        )
            except BudgetExceeded:
                for future in futures:
                    future.cancel()
                raise
        return [result or Result("", 0, 0, 0, cache_hit=False, error="cancelled") for result in results]

    def summary(self) -> dict:
        return {
            "model": self.model,
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "failures": self.failures,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "spent_cny": round(self.spent_cny, 4),
            "budget_cny": self.budget_cny,
            "per_stage": {
                stage: {"calls": int(bucket["calls"]), "cost_cny": round(bucket["cost_cny"], 4)}
                for stage, bucket in sorted(self.per_stage.items())
            },
        }

    def write_spend_line(self, stage: str) -> Path:
        """Append one line of actual usage to ``reports/SPEND.md``.

        Only counts are recorded. The authoritative cost figure is the balance
        delta the human can read with `python run.py spend`; guessing prices into
        a file would just add a number nobody should trust.
        """
        target = repo_path("reports", "SPEND.md")
        target.parent.mkdir(parents=True, exist_ok=True)
        summary = self.summary()
        line = (
            f"| {stage} | {summary['calls']} | {summary['cache_hits']} | "
            f"{summary['prompt_tokens']} | {summary['completion_tokens']} |"
        )
        if not target.exists():
            target.write_text(
                "| stage | api_calls | cache_hits | prompt_tokens | completion_tokens |\n"
                "|---|---|---|---|---|\n",
                encoding="utf-8",
                newline="\n",
            )
        with target.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
        return target
