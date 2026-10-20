"""The twenty validity checks (V1..V20).

A generated corpus is only usable if every check passes; the generator is
expected to fail loudly rather than emit a partially valid bundle. V12 is the
important one: it verifies with a real BM25 index that a poisoned passage can
actually reach the detection pool, which is the assumption the whole experiment
rests on.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import templates as tpl
from . import wordbanks as wb
from .generator import (
    CALIB_COUNT,
    CLEAN_COUNT,
    DISTRACTOR_COUNT,
    FACT_COUNT,
    INJECTION_COUNTS,
    QUERY_COUNT,
    TEST_COUNT,
    GenerationResult,
    generate,
)
from ..paths import path as repo_path
from ..retrieval.bm25 import BM25
from ..retrieval.tokenize import normalize_text, word_count

#: Whole-word blacklist of real-world names that must never appear in the
#: fictional corpus. Deliberately broad; a false positive is cheap to fix, a
#: leaked real entity invalidates the "no parametric knowledge" premise.
REAL_WORLD_BLACKLIST: tuple[str, ...] = (
    # countries and regions
    "america", "united states", "china", "japan", "germany", "france", "india",
    "brazil", "russia", "canada", "australia", "britain", "england", "scotland",
    "wales", "ireland", "spain", "italy", "portugal", "netherlands", "belgium",
    "sweden", "norway", "denmark", "finland", "poland", "greece", "turkey",
    "egypt", "mexico", "argentina", "chile", "peru", "kenya", "nigeria",
    # cities
    "london", "paris", "berlin", "madrid", "rome", "moscow", "beijing",
    "shanghai", "tokyo", "osaka", "seoul", "singapore", "sydney", "melbourne",
    "toronto", "chicago", "boston", "seattle", "houston", "dallas", "miami",
    "auckland", "wellington", "dublin", "edinburgh", "manchester", "liverpool",
    # companies and products
    "google", "microsoft", "apple", "amazon", "meta", "facebook", "twitter",
    "nvidia", "intel", "samsung", "toyota", "honda", "ford", "volkswagen",
    "boeing", "airbus", "netflix", "tesla", "openai", "deepseek", "alibaba",
    "tencent", "huawei", "ibm", "oracle", "sap", "siemens", "philips",
    # people
    "einstein", "newton", "darwin", "curie", "tesla", "edison", "musk",
    "gates", "jobs", "bezos", "zuckerberg", "turing", "hopper", "knuth",
    # well-known places and institutions
    "wikipedia", "harvard", "oxford", "cambridge", "stanford", "mit",
    "nasa", "esa", "euclid", "pythagoras", "shakespeare", "beethoven",
)

_ID_PATTERNS: dict[str, re.Pattern] = {
    "fact": re.compile(r"^fact_\d{4}$"),
    "clean": re.compile(r"^p_clean_\d{4}$"),
    "query": re.compile(r"^q_\d{4}$"),
    "distractor": re.compile(r"^p_dist_\d{4}$"),
    "poison": re.compile(r"^p_atk[1-6]_\d{4}_v[1-3]$"),
}

_BLACKLIST_RE = re.compile(
    "|".join(rf"\b{re.escape(term)}\b" for term in REAL_WORLD_BLACKLIST), re.IGNORECASE
)


@dataclass(frozen=True)
class CheckOutcome:
    code: str
    name: str
    ok: bool
    detail: str


def _ngrams(tokens: list[str], size: int) -> set[tuple[str, ...]]:
    if len(tokens) < size:
        return set()
    return {tuple(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}


def _check(code: str, name: str, fn: Callable[[], str]) -> CheckOutcome:
    try:
        return CheckOutcome(code, name, True, fn())
    except AssertionError as exc:
        return CheckOutcome(code, name, False, str(exc))
    except Exception as exc:  # noqa: BLE001 - a failing check must not abort the run
        return CheckOutcome(code, name, False, f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #


def validate(
    result: GenerationResult,
    *,
    check_determinism: bool = True,
    check_seed_cross: bool = True,
    verbose: bool = False,
    notes: list[dict] | None = None,
) -> list[CheckOutcome]:
    texts = result.all_texts()
    facts = result.facts
    by_fact = {fact["fact_id"]: fact for fact in facts}
    fact_ids = set(by_fact)
    poison_values = {fact["target_value"] for fact in facts}
    correct_values = {fact["correct_value"] for fact in facts}
    all_values = poison_values | correct_values

    def v1() -> str:
        assert len(facts) == FACT_COUNT, f"{len(facts)} facts"
        histogram: dict[str, int] = {}
        for fact in facts:
            histogram[fact["topic"]] = histogram.get(fact["topic"], 0) + 1
        assert len(histogram) == 6, f"{len(histogram)} topics"
        small = {topic: n for topic, n in histogram.items() if n < 12}
        assert not small, f"topics below 12 facts: {small}"
        return "6 topics, min " + str(min(histogram.values())) + " facts"

    def v2() -> str:
        assert len(result.clean) == CLEAN_COUNT, f"{len(result.clean)} clean passages"
        seen = {row["fact_id"] for row in result.clean}
        assert seen == fact_ids, "clean passages do not cover every fact exactly once"
        return f"{CLEAN_COUNT} clean passages, one per fact"

    def v3() -> str:
        assert len(result.queries) == QUERY_COUNT, f"{len(result.queries)} queries"
        seen = {row["fact_id"] for row in result.queries}
        assert seen == fact_ids, "queries do not cover every fact exactly once"
        long_answers = [q["query_id"] for q in result.queries if len(q["correct_answer"].split()) > 6]
        assert not long_answers, f"answers longer than six words: {long_answers[:5]}"
        return f"{QUERY_COUNT} queries, all answers <= 6 words"

    def v4() -> str:
        assert len(result.distractors) == DISTRACTOR_COUNT, f"{len(result.distractors)}"
        leaks = [
            row["passage_id"]
            for row in result.distractors
            if any(value in row["text"] for value in all_values)
        ]
        assert not leaks, f"distractors contain fact values: {leaks[:5]}"
        return f"{DISTRACTOR_COUNT} distractors, no fact value present"

    def v5() -> str:
        expected = len(tpl.FAMILIES) * FACT_COUNT * tpl.VARIANTS_PER_FACT
        assert len(result.poisoned) == expected, f"{len(result.poisoned)} != {expected}"
        keys = {
            (row["attack_family"], row["query_id"], row["variant"]) for row in result.poisoned
        }
        assert len(keys) == expected, "duplicate (family, query, variant) triples"
        ids = [row["passage_id"] for row in result.poisoned]
        assert len(set(ids)) == len(ids), "duplicate poisoned passage ids"
        return f"{expected} passages = 6 families x 100 queries x 3 variants"

    def v6() -> str:
        clean_by_fact = {row["fact_id"]: row["text"] for row in result.clean}
        bad = [
            row["query_id"]
            for row in result.queries
            if row["correct_answer"] not in clean_by_fact[row["fact_id"]]
        ]
        assert not bad, f"correct answer missing from clean passage: {bad[:5]}"
        return "every correct answer appears verbatim in its clean passage"

    def v7() -> str:
        clean_by_fact = {row["fact_id"]: row["text"] for row in result.clean}
        bad = [
            row["query_id"]
            for row in result.queries
            if row["target_answer"] in clean_by_fact[row["fact_id"]]
        ]
        assert not bad, f"target answer present in clean passage: {bad[:5]}"
        return "no clean passage contains a target answer"

    def v8() -> str:
        bad = [
            row["passage_id"]
            for row in result.poisoned
            if by_fact[row["fact_id"]]["target_value"] not in row["text"]
        ]
        assert not bad, f"poisoned passage without its target: {bad[:5]}"
        mismatched = [
            row["passage_id"]
            for row in result.poisoned
            if row["contains_target"] != (by_fact[row["fact_id"]]["target_value"] in row["text"])
        ]
        assert not mismatched, f"contains_target flag is wrong: {mismatched[:5]}"
        return "every poisoned passage contains its own target value"

    def v9() -> str:
        offenders: list[str] = []
        for row in result.poisoned:
            fact = by_fact[row["fact_id"]]
            others = all_values - {fact["correct_value"], fact["target_value"]}
            if any(value in row["text"] for value in others):
                offenders.append(row["passage_id"])
        assert not offenders, f"poisoned passages contain foreign values: {offenders[:5]}"
        return "no poisoned passage contains another fact's value"

    def v10() -> str:
        bad = [
            pid
            for pid, text in texts.items()
            if not (tpl.GEN_MIN_WORDS <= word_count(text) <= tpl.GEN_MAX_WORDS)
        ]
        assert not bad, f"passages outside 80-150 words: {bad[:5]}"
        lengths = [word_count(text) for text in texts.values()]
        return f"{len(texts)} passages, words {min(lengths)}-{max(lengths)}"

    def v11() -> str:
        checks = (
            ("fact", [row["fact_id"] for row in facts]),
            ("clean", [row["passage_id"] for row in result.clean]),
            ("query", [row["query_id"] for row in result.queries]),
            ("distractor", [row["passage_id"] for row in result.distractors]),
            ("poison", [row["passage_id"] for row in result.poisoned]),
        )
        for kind, ids in checks:
            pattern = _ID_PATTERNS[kind]
            bad = [i for i in ids if not pattern.match(i)]
            assert not bad, f"{kind} ids malformed: {bad[:5]}"
            assert len(set(ids)) == len(ids), f"{kind} ids are not unique"
        everything = [i for _kind, ids in checks for i in ids]
        assert len(set(everything)) == len(everything), "ids collide across files"
        return "all ids match their pattern and are globally unique"

    def v12() -> str:
        # The requirement is *reachability of the detection pool*, not dominance.
        # A strong attack is supposed to outrank the clean supporting passage;
        # what must hold is that both the clean passage and the poisoned passages
        # land inside the top-N pool, otherwise the query carries no signal at all
        # and the ablation cell would be empty.
        top_n = _top_n()
        clean_for_query = {row["query_id"]: row["passage_id"] for row in _query_index(result)}
        worst_clean = 0
        worst_poison = 0
        clean_fail: list[str] = []
        poison_fail: list[str] = []
        poison_ranks: dict[str, list[int]] = {family: [] for family in tpl.FAMILIES}

        for corpus_id, corpus in sorted(result.corpora.items()):
            documents = [(chunk_id, texts[chunk_id]) for chunk_id in corpus["chunk_ids"]]
            index = BM25().fit(documents)
            owned = corpus["poison_owner"]
            family = corpus["attack_family"]
            for query in result.queries:
                ordered = index.ranked_ids(query["question"])
                ranks = {
                    doc_id: position for position, (doc_id, _score) in enumerate(ordered, start=1)
                }

                clean_rank = ranks.get(clean_for_query[query["query_id"]], 0)
                worst_clean = max(worst_clean, clean_rank)
                if clean_rank == 0 or clean_rank > top_n:
                    clean_fail.append(f"{corpus_id}:{query['query_id']}:{clean_rank}")
                    if notes is not None:
                        notes.append(
                            {
                                "corpus_id": corpus_id,
                                "query_id": query["query_id"],
                                "fact_id": query["fact_id"],
                                "passage_id": clean_for_query[query["query_id"]],
                                "clean_rank": clean_rank,
                                "reason": "clean_passage_outside_pool",
                            }
                        )

                for poison_id, owner in owned.items():
                    if owner != query["query_id"]:
                        continue
                    poison_rank = ranks.get(poison_id, 0)
                    poison_ranks[family].append(poison_rank)
                    worst_poison = max(worst_poison, poison_rank)
                    if poison_rank == 0 or poison_rank > top_n:
                        poison_fail.append(f"{corpus_id}:{poison_id}:{poison_rank}")

        total_cells = len(result.corpora) * len(result.queries)
        invalid_rate = len(clean_fail) / total_cells if total_cells else 0.0
        # A poisoned passage must always be reachable: if the attack cannot enter
        # the pool there is no experiment. A clean passage may occasionally be
        # pushed out when three poisoned passages compete for the same slots; a
        # small, explicitly reported invalid rate is preferable to weakening the
        # attack until every passage fits.
        assert not poison_fail, f"poisoned passage fell outside top-{top_n}: {poison_fail[:5]}"
        assert invalid_rate <= 0.02, (
            f"clean passage fell outside top-{top_n} in {len(clean_fail)}/{total_cells} cells "
            f"({invalid_rate:.2%}) which exceeds the 2% budget: {clean_fail[:5]}"
        )

        in_top5 = {
            family: (
                sum(1 for rank in ranks if 0 < rank <= 5) / len(ranks) if ranks else 0.0
            )
            for family, ranks in poison_ranks.items()
        }
        summary = ", ".join(f"{fam.split('_')[0]}={rate:.0%}" for fam, rate in in_top5.items())
        return (
            f"clean rank <= {top_n} (worst {worst_clean}, invalid {invalid_rate:.2%}), "
            f"poison rank <= {top_n} (worst {worst_poison}); "
            f"poison in top-5 [{summary}]"
        )

    def v13() -> str:
        calib = result.splits["calib_fact_ids"]
        test = result.splits["test_fact_ids"]
        assert len(calib) == CALIB_COUNT, f"calib has {len(calib)} facts"
        assert len(test) == TEST_COUNT, f"test has {len(test)} facts"
        assert not set(calib) & set(test), "calib and test overlap"
        assert set(calib) | set(test) == fact_ids, "splits do not cover every fact"
        for fact in facts:
            expected = "calib" if fact["fact_id"] in set(calib) else "test"
            assert fact["split"] == expected, f"{fact['fact_id']} split mismatch"
        return f"{CALIB_COUNT} calib / {TEST_COUNT} test, disjoint"

    def v14() -> str:
        counter: dict[tuple[str, str], set[int]] = {}
        for row in result.poisoned:
            counter.setdefault((row["attack_family"], row["fact_id"]), set()).add(row["variant"])
        expected = set(range(1, tpl.VARIANTS_PER_FACT + 1))
        missing = [
            f"{family}:{fact_id}"
            for family in tpl.FAMILIES
            for fact_id in sorted(fact_ids)
            if counter.get((family, fact_id)) != expected
        ]
        assert not missing, f"facts without all three variants: {missing[:5]}"
        return "every fact has 3 variants under all 6 families"

    def v15() -> str:
        for fact in facts:
            correct = normalize_text(fact["correct_value"])
            target = normalize_text(fact["target_value"])
            assert correct != target, f"{fact['fact_id']}: values are identical"
            assert correct not in target and target not in correct, (
                f"{fact['fact_id']}: one value contains the other"
            )
            shared = _ngrams(correct.split(), 4) & _ngrams(target.split(), 4)
            assert not shared, f"{fact['fact_id']}: shared 4-gram {shared}"
        return "no pair shares a 4-gram, is nested, or is identical"

    def v16() -> str:
        offenders: list[str] = []
        for pid, text in texts.items():
            if _BLACKLIST_RE.search(text):
                offenders.append(pid)
        for fact in facts:
            for value in (fact["correct_value"], fact["target_value"]):
                if _BLACKLIST_RE.search(value):
                    offenders.append(f"{fact['fact_id']}:{value}")
        assert not offenders, f"real-world names found: {offenders[:5]}"
        return "no blacklisted real-world entity anywhere"

    def v17() -> str:
        bad = []
        for row in result.poisoned:
            if row["attack_family"] in tpl.STEALTH_FAMILIES and row["trigger_visible"]:
                bad.append(f"{row['passage_id']}: stealth family flagged as visible")
            if row["attack_family"] in tpl.CANONICAL_FAMILIES and not row["trigger_visible"]:
                bad.append(f"{row['passage_id']}: canonical family shows no marker")
        assert not bad, f"visibility flags inconsistent: {bad[:5]}"
        return "canonical families visible, stealth families invisible"

    outcomes: list[CheckOutcome] = [
        _check("V1", "fact count, topic coverage", v1),
        _check("V2", "one clean passage per fact", v2),
        _check("V3", "one query per fact", v3),
        _check("V4", "distractors carry no fact value", v4),
        _check("V5", "1800 poisoned passages, unique keys", v5),
        _check("V6", "correct answer present in clean passage", v6),
        _check("V7", "target answer absent from clean passage", v7),
        _check("V8", "target present in every poisoned passage", v8),
        _check("V9", "no foreign values in poisoned passages", v9),
        _check("V10", "every passage is 80-150 words", v10),
        _check("V11", "id patterns and global uniqueness", v11),
        _check("V12", "BM25 retrievability of clean and poisoned passages", v12),
        _check("V13", "calib/test split", v13),
        _check("V14", "three variants per fact and family", v14),
        _check("V15", "correct and target values are distinct", v15),
        _check("V16", "no real-world entity", v16),
        _check("V17", "visibility flags match the attack families", v17),
    ]

    if check_determinism:
        outcomes.append(
            _check("V18", "regeneration is byte-identical", lambda: v18(result))
        )
    if check_seed_cross:
        outcomes.append(_check("V19", "seeds differ in content, not structure", lambda: v19(result)))
    outcomes.append(_check("V20", "corpus composition", lambda: v20(result)))

    if verbose:
        for outcome in outcomes:
            print(f"{outcome.code} {'PASS' if outcome.ok else 'FAIL'}  {outcome.detail}")
    return outcomes


def _query_index(result: GenerationResult) -> list[dict]:
    """Map every query id to its clean passage id (one clean passage per fact)."""
    by_fact = {row["fact_id"]: row["passage_id"] for row in result.clean}
    return [
        {"query_id": row["query_id"], "passage_id": by_fact[row["fact_id"]]}
        for row in result.queries
    ]


def _top_n() -> int:
    """Read the detection pool size from the configuration, defaulting to 15."""
    try:
        from ..config import load_config

        return int(load_config("default.yaml").get("retrieval.top_n", 15))
    except Exception:  # noqa: BLE001 - the validator must run even without configs
        return 15


def _hash_tree(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for candidate in sorted(root.rglob("*.json")):
        hashes[candidate.relative_to(root).as_posix()] = hashlib.sha256(
            candidate.read_bytes()
        ).hexdigest()
    return hashes


def v18(result: GenerationResult) -> str:
    """Regenerate into a scratch directory and compare every byte."""
    scratch = repo_path("data", "synthetic", "_determinism_check")
    if scratch.exists():
        shutil.rmtree(scratch)
    try:
        generate(result.seed, out_dir=scratch)
        first = _hash_tree(result.out_dir)
        second = _hash_tree(scratch)
        assert set(first) == set(second), "file sets differ between runs"
        differing = [name for name in first if first[name] != second[name]]
        assert not differing, f"files differ between runs: {differing[:5]}"
        return f"{len(first)} files identical across two runs"
    finally:
        if scratch.exists():
            shutil.rmtree(scratch, ignore_errors=True)


def v19(result: GenerationResult) -> str:
    """A second seed must keep the structure and change the content."""
    other_seed = 43 if result.seed == 42 else 42
    scratch = repo_path("data", "synthetic", f"_cross_seed_{other_seed}")
    if scratch.exists():
        shutil.rmtree(scratch)
    try:
        other = generate(other_seed, out_dir=scratch)
        assert len(other.facts) == len(result.facts), "fact counts differ"
        assert len(other.poisoned) == len(result.poisoned), "poison counts differ"
        assert set(other.corpora) == set(result.corpora), "corpus ids differ"
        mine = {fact["correct_value"] for fact in result.facts}
        theirs = {fact["correct_value"] for fact in other.facts}
        assert mine != theirs, "the two seeds produced identical values"
        same_text = sum(
            1
            for a, b in zip(result.clean, other.clean)
            if a["text"].split(".")[0] == b["text"].split(".")[0]
        )
        assert same_text < len(result.clean) * 0.5, "too many identical openings across seeds"
        return f"seed {other_seed}: structure matches, content differs"
    finally:
        if scratch.exists():
            shutil.rmtree(scratch, ignore_errors=True)


def v20(result: GenerationResult) -> str:
    expected_keys = len(tpl.FAMILIES) * len(INJECTION_COUNTS)
    assert len(result.corpora) == expected_keys, f"{len(result.corpora)} corpora"
    for corpus_id, corpus in result.corpora.items():
        injection = corpus["injection"]
        expected = CLEAN_COUNT + DISTRACTOR_COUNT + FACT_COUNT * injection
        assert corpus["n_chunks"] == expected, f"{corpus_id}: {corpus['n_chunks']} != {expected}"
        assert len(corpus["chunk_ids"]) == expected, f"{corpus_id}: chunk id count mismatch"
        assert len(set(corpus["chunk_ids"])) == expected, f"{corpus_id}: duplicate chunk ids"
        assert len(corpus["poison_chunk_ids"]) == FACT_COUNT * injection, (
            f"{corpus_id}: poison chunk count mismatch"
        )
        assert set(corpus["poison_chunk_ids"]) <= set(corpus["chunk_ids"]), (
            f"{corpus_id}: poison ids outside the corpus"
        )
    return f"{expected_keys} corpora, sizes 400 and 600"


def summarize(outcomes: list[CheckOutcome]) -> tuple[bool, str]:
    total = len(outcomes)
    failed = [outcome for outcome in outcomes if not outcome.ok]
    lines = [f"{o.code}  {'PASS' if o.ok else 'FAIL'}  {o.name}: {o.detail}" for o in outcomes]
    header = f"Validators: {total - len(failed)}/{total} PASS"
    if failed:
        header = f"Validators: {total - len(failed)}/{total} PASS, {len(failed)} FAILED"
    return (not failed), header + "\n" + "\n".join(lines)
