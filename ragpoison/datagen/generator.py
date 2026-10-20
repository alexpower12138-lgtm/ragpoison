"""Deterministic synthetic corpus generator.

Everything is derived from one integer seed: there is no clock, no API call and
no set/dict iteration order dependence in the output. Running the generator
twice with the same seed produces byte-identical files, which is checked by
validator V18 and recorded as SHA-256 in ``manifest.json``.

Note on timestamps: no file contains a generation time. A timestamp would make
byte-identical regeneration impossible, which is exactly the property the
benchmark needs.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import templates as tpl
from . import wordbanks as wb
from ..paths import path as repo_path
from ..retrieval.tokenize import word_count

GENERATOR_VERSION = "1.0.0"
FACT_COUNT = 100
CLEAN_COUNT = 100
QUERY_COUNT = 100
DISTRACTOR_COUNT = 200
CALIB_COUNT = 30
TEST_COUNT = 70
INJECTION_COUNTS = (1, 3)

JSON_KW: dict[str, Any] = {"indent": 2, "ensure_ascii": False, "sort_keys": False}


def rng_for(seed: int, key: str) -> random.Random:
    """A random source that depends only on (seed, key), never on call order."""
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


@dataclass
class GenerationResult:
    seed: int
    out_dir: Path
    facts: list[dict] = field(default_factory=list)
    clean: list[dict] = field(default_factory=list)
    queries: list[dict] = field(default_factory=list)
    distractors: list[dict] = field(default_factory=list)
    poisoned: list[dict] = field(default_factory=list)
    splits: dict = field(default_factory=dict)
    corpora: dict[str, dict] = field(default_factory=dict)
    sha256: dict[str, str] = field(default_factory=dict)

    def counts(self) -> dict[str, int]:
        return {
            "facts": len(self.facts),
            "clean_passages": len(self.clean),
            "queries": len(self.queries),
            "distractor_passages": len(self.distractors),
            "poisoned_passages": len(self.poisoned),
            "corpora": len(self.corpora),
        }

    def all_texts(self) -> dict[str, str]:
        """Every passage id mapped to its text, used by the validators."""
        mapping: dict[str, str] = {}
        for row in self.clean:
            mapping[row["passage_id"]] = row["text"]
        for row in self.distractors:
            mapping[row["passage_id"]] = row["text"]
        for row in self.poisoned:
            mapping[row["passage_id"]] = row["text"]
        return mapping


# --------------------------------------------------------------------------- #
# JSON output
# --------------------------------------------------------------------------- #


def dump_json(payload: Any) -> str:
    """Serialise to a stable, newline-terminated UTF-8 string."""
    return json.dumps(payload, **JSON_KW) + "\n"


def write_json(target: Path, payload: Any) -> str:
    """Write ``payload`` and return the SHA-256 of the bytes actually written."""
    text = dump_json(payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Fact construction
# --------------------------------------------------------------------------- #


def _entity_pool(topic: str, seed: int) -> list[str]:
    prefixes, suffixes = wb.TOPIC_ENTITIES[topic]
    combos = [f"{prefix} {suffix}" for prefix in prefixes for suffix in suffixes]
    rng = rng_for(seed, f"entities:{topic}")
    rng.shuffle(combos)
    return combos


def build_facts(seed: int) -> tuple[list[dict], dict[str, str]]:
    """Return the 100 facts plus the topic of every fact id."""
    used_values: set[str] = set()
    facts: list[dict] = []
    topics: dict[str, str] = {}
    counter = 0

    for topic, plan in tpl.TOPIC_PLAN.items():
        pool = _entity_pool(topic, seed)
        cursor = 0
        for attribute_key, count in plan:
            attribute = tpl.ATTRIBUTES[attribute_key]
            for _ in range(count):
                counter += 1
                fact_id = f"fact_{counter:04d}"
                entity = pool[cursor]
                cursor += 1
                rng = rng_for(seed, f"values:{fact_id}")
                correct = tpl.allocate_value(attribute, rng, used_values)
                # The target is a plausible neighbour of the correct value, so
                # that the attack cannot be rejected by an implausible number.
                target = tpl.allocate_value(attribute, rng, used_values, near=correct)
                question = attribute.question(entity, rng.randrange(2))
                fact = {
                    "fact_id": fact_id,
                    "entity": entity,
                    "attribute_key": attribute_key,
                    "attribute": attribute.label,
                    "answer_type": attribute.answer_type,
                    "correct_value": correct,
                    "target_value": target,
                    "topic": topic,
                    "split": "",  # filled in below
                    "question": question,
                }
                facts.append(fact)
                topics[fact_id] = topic

    if len(facts) != FACT_COUNT:  # pragma: no cover - guards the topic plan
        raise RuntimeError(f"expected {FACT_COUNT} facts, built {len(facts)}")
    return facts, topics


def assign_splits(facts: list[dict], seed: int) -> dict[str, list[str]]:
    """Deterministically split the facts into calib (30) and test (70)."""
    rng = rng_for(seed, "splits")
    order = [fact["fact_id"] for fact in facts]
    rng.shuffle(order)
    calib = sorted(order[:CALIB_COUNT])
    test = sorted(order[CALIB_COUNT:])
    if len(calib) != CALIB_COUNT or len(test) != TEST_COUNT:  # pragma: no cover
        raise RuntimeError("split sizes are wrong")
    by_id = {fact["fact_id"]: fact for fact in facts}
    for fact_id in calib:
        by_id[fact_id]["split"] = "calib"
    for fact_id in test:
        by_id[fact_id]["split"] = "test"
    return {"calib_fact_ids": calib, "test_fact_ids": test}


# --------------------------------------------------------------------------- #
# Passage construction
# --------------------------------------------------------------------------- #


def build_clean_and_queries(
    facts: list[dict], topics: dict[str, str], seed: int
) -> tuple[list[dict], list[dict]]:
    clean: list[dict] = []
    queries: list[dict] = []
    for index, fact in enumerate(facts, start=1):
        rng = rng_for(seed, f"clean:{fact['fact_id']}")
        topic = topics[fact["fact_id"]]
        text = tpl.build_clean(fact, topic, rng)
        clean.append(
            {
                "passage_id": f"p_clean_{index:04d}",
                "fact_id": fact["fact_id"],
                "text": text,
                "supports_answer": fact["correct_value"],
                "n_words": word_count(text),
            }
        )
        queries.append(
            {
                "query_id": f"q_{index:04d}",
                "fact_id": fact["fact_id"],
                "question": fact["question"],
                "correct_answer": fact["correct_value"],
                "target_answer": fact["target_value"],
                "answer_type": fact["answer_type"],
                "topic": topic,
                "split": fact["split"],
            }
        )
    return clean, queries


def build_distractors(seed: int) -> list[dict]:
    """200 topically plausible passages that contain no fact value."""
    topic_order = list(wb.TOPIC_DONORS.keys())
    cursors: dict[str, list[str]] = {}
    for topic, (prefixes, suffixes) in wb.TOPIC_DONORS.items():
        combos = [f"{prefix} {suffix}" for prefix in prefixes for suffix in suffixes]
        rng = rng_for(seed, f"donors:{topic}")
        rng.shuffle(combos)
        cursors[topic] = combos

    distractors: list[dict] = []
    used: dict[str, int] = {topic: 0 for topic in topic_order}
    for index in range(1, DISTRACTOR_COUNT + 1):
        topic = topic_order[(index - 1) % len(topic_order)]
        pool = cursors[topic]
        donor = pool[used[topic] % len(pool)]
        used[topic] += 1
        rng = rng_for(seed, f"distractor:{index:04d}")
        text = tpl.build_distractor(topic, donor, rng)
        distractors.append(
            {
                "passage_id": f"p_dist_{index:04d}",
                "topic": topic,
                "donor_entity": donor,
                "text": text,
                "related_fact_ids": [],
                "n_words": word_count(text),
            }
        )
    return distractors


def build_poisoned(facts: list[dict], topics: dict[str, str], seed: int) -> list[dict]:
    poisoned: list[dict] = []
    for family in tpl.FAMILIES:
        short = family.split("_")[0].lower()  # atk1 .. atk6
        for index, fact in enumerate(facts, start=1):
            topic = topics[fact["fact_id"]]
            for variant in range(1, tpl.VARIANTS_PER_FACT + 1):
                rng = rng_for(seed, f"poison:{family}:{fact['fact_id']}:{variant}")
                fact_with_question = fact
                text = tpl.build_poison(family, variant, fact_with_question, topic, rng)
                poisoned.append(
                    {
                        "passage_id": f"p_{short}_{index:04d}_v{variant}",
                        "attack_family": family,
                        "variant": variant,
                        "query_id": f"q_{index:04d}",
                        "fact_id": fact["fact_id"],
                        "text": text,
                        "contains_target": fact["target_value"] in text,
                        "contains_correct": fact["correct_value"] in text,
                        "trigger_visible": tpl.visible_markers(text, fact["question"], family),
                        "n_words": word_count(text),
                    }
                )
    return poisoned


# --------------------------------------------------------------------------- #
# Corpora
# --------------------------------------------------------------------------- #


def build_corpora(
    clean: list[dict], distractors: list[dict], poisoned: list[dict]
) -> dict[str, dict]:
    """One corpus per (attack family, injection count)."""
    base_ids = [row["passage_id"] for row in clean] + [row["passage_id"] for row in distractors]
    by_family: dict[str, list[dict]] = {}
    for row in poisoned:
        by_family.setdefault(row["attack_family"], []).append(row)

    corpora: dict[str, dict] = {}
    for family in tpl.FAMILIES:
        rows = sorted(by_family[family], key=lambda r: r["passage_id"])
        for injection in INJECTION_COUNTS:
            selected = [row for row in rows if row["variant"] <= injection]
            poison_ids = [row["passage_id"] for row in selected]
            owner = {row["passage_id"]: row["query_id"] for row in selected}
            corpus_id = f"{family}_inj{injection}"
            corpora[corpus_id] = {
                "corpus_id": corpus_id,
                "attack_family": family,
                "injection": injection,
                "chunk_ids": base_ids + poison_ids,
                "poison_chunk_ids": poison_ids,
                "poison_owner": owner,
                "n_chunks": len(base_ids) + len(poison_ids),
            }
    return corpora


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def generate(seed: int, out_dir: Path | None = None) -> GenerationResult:
    """Build the whole bundle for ``seed`` and write it to disk."""
    target = out_dir if out_dir is not None else repo_path("data", "synthetic", f"seed{seed}")
    target.mkdir(parents=True, exist_ok=True)

    facts, topics = build_facts(seed)
    splits = assign_splits(facts, seed)
    clean, queries = build_clean_and_queries(facts, topics, seed)
    distractors = build_distractors(seed)
    poisoned = build_poisoned(facts, topics, seed)
    corpora = build_corpora(clean, distractors, poisoned)

    result = GenerationResult(
        seed=seed,
        out_dir=target,
        facts=facts,
        clean=clean,
        queries=queries,
        distractors=distractors,
        poisoned=poisoned,
        splits={**splits, "calib": CALIB_COUNT, "test": TEST_COUNT},
        corpora=corpora,
    )

    facts_payload = [
        {
            "fact_id": fact["fact_id"],
            "entity": fact["entity"],
            "attribute": fact["attribute"],
            "correct_value": fact["correct_value"],
            "target_value": fact["target_value"],
            "topic": fact["topic"],
            "answer_type": fact["answer_type"],
            "split": fact["split"],
        }
        for fact in facts
    ]

    hashes: dict[str, str] = {}
    hashes["facts.json"] = write_json(target / "facts.json", facts_payload)
    hashes["clean_passages.json"] = write_json(target / "clean_passages.json", clean)
    hashes["queries.json"] = write_json(target / "queries.json", queries)
    hashes["distractor_passages.json"] = write_json(target / "distractor_passages.json", distractors)
    hashes["poisoned_passages.json"] = write_json(target / "poisoned_passages.json", poisoned)
    hashes["splits.json"] = write_json(target / "splits.json", result.splits)
    for corpus_id, payload in corpora.items():
        hashes[f"corpora/{corpus_id}.json"] = write_json(
            target / "corpora" / f"{corpus_id}.json", payload
        )

    manifest = {
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "counts": result.counts(),
        "injection_counts": list(INJECTION_COUNTS),
        "variants_per_fact": tpl.VARIANTS_PER_FACT,
        "families": list(tpl.FAMILIES),
        "calib": CALIB_COUNT,
        "test": TEST_COUNT,
        # Every generated file except the manifest itself, so that the manifest
        # never has to contain its own hash (which is impossible to satisfy).
        "sha256": hashes,
    }
    write_json(target / "manifest.json", manifest)

    result.sha256 = hashes
    return result


def topic_histogram(facts: list[dict]) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for fact in facts:
        histogram[fact["topic"]] = histogram.get(fact["topic"], 0) + 1
    return dict(sorted(histogram.items()))
