"""Calibration statistics for the local signals.

S2 and S6 need to know what "normal" looks like, and they must learn it from data
the experiment never scores:

* only queries in the **calibration split** are used (30 of the 100 facts);
* only **benign** pools are used: the base pool with no poisoned chunk in it.

The result is cached to ``cache/calibration_seed<N>.json`` and is a pure function
of the generated bundle, so it can be rebuilt at any time and compared byte for
byte.
"""

from __future__ import annotations

import json

import numpy as np

from ..datagen.corpora import base_pool
from ..datagen.loader import Bundle, load_bundle
from ..paths import path as repo_path
from ..retrieval.bm25 import BM25
from ..retrieval.tfidf import TfidfIndex
from . import s6_pseudo_ppl

CALIBRATION_VERSION = 1


def calibration_path(seed: int):
    return repo_path("cache", f"calibration_seed{seed}.json")


def compute(seed: int, *, force: bool = False, top_n: int = 15) -> dict:
    """Build (or load) the calibration statistics for ``seed``."""
    target = calibration_path(seed)
    if target.exists() and not force:
        try:
            cached = json.loads(target.read_text(encoding="utf-8"))
            if cached.get("version") == CALIBRATION_VERSION:
                return cached
        except (OSError, ValueError):
            pass

    bundle: Bundle = load_bundle(seed)
    texts = bundle.texts()
    pool = base_pool(bundle)
    calib_ids = set(bundle.splits["calib_fact_ids"])
    calib_queries = [row for row in bundle.queries if row["fact_id"] in calib_ids]

    bm25 = BM25().fit([(chunk_id, texts[chunk_id]) for chunk_id in pool])
    tfidf = TfidfIndex().fit([(chunk_id, texts[chunk_id]) for chunk_id in pool])

    # S2: mean cross-similarity among the candidates of benign pools.
    means: list[float] = []
    for query in calib_queries:
        retrieved = [chunk_id for chunk_id, _score in bm25.ranked_ids(query["question"], top_n=top_n)]
        means.extend(tfidf.mean_similarity_to_others(retrieved).values())
    array = np.asarray(means, dtype=float)
    s2 = {
        "mu": float(array.mean()),
        "sigma": float(array.std(ddof=0)),
        "n_values": int(array.size),
        "n_pools": len(calib_queries),
    }

    # S6: the character n-gram model and the NLL distribution of benign text.
    benign_texts = [texts[chunk_id] for chunk_id in pool]
    s6 = s6_pseudo_ppl.train(benign_texts)

    payload = {
        "version": CALIBRATION_VERSION,
        "seed": seed,
        "split": "calib",
        "n_calib_queries": len(calib_queries),
        "n_benign_texts": len(benign_texts),
        "top_n": top_n,
        "s2": s2,
        "s6": s6,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return payload
