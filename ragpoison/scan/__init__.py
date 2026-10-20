"""Periodic corpus audit, PCA (WP9).

Planned modules:
    policy.py      scan policy resolution, priority queue, budget accounting
    coarse.py      stage-1 coarse filter (no LLM)
    batch_scan.py  stage-2 batched LLM scan with JSON-array output
    store.py       chunk_status bookkeeping, dirty set, scan versions
    scheduler.py   manual and cron-triggered runs
"""

from __future__ import annotations
