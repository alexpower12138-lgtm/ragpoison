"""Corpus composition: the two designs the project needs.

The first version of this benchmark put every poisoned passage into one shared
knowledge base and evaluated all 100 queries against it. The retrieval audit
showed why that is wrong for query-time detection: because all 100 questions
share a sentence template, a poisoned passage written for query A also scores
highly for query B, so a retrieval pool fills up with poisoned text that is not
an attack on the query being evaluated. A detector that correctly identifies
"this is poisoned text" is then scored as producing many false positives.

Two designs, one per research question:

``query_corpus`` (RQ1, query-time detection)
    Base pool plus the poisoned passages written for *this* query only. This is
    the faithful PoisonedRAG setting: a handful of poisoned chunks inside a
    larger clean base. The pool then contains the attack, the clean evidence and
    ordinary neighbouring passages, which is what the detector is asked to
    separate.

``shared_corpora`` (RQ2, periodic corpus audit)
    Base pool plus every poisoned passage of one attack family. This is the case
    the corpus audit exists for: the knowledge base really does contain a hundred
    poisoned chunks and the job is to find them on a budget. Contamination of a
    retrieval pool is the phenomenon being measured there, not a defect.
"""

from __future__ import annotations

from typing import Iterable, Protocol


class CorpusSource(Protocol):
    """Anything that carries the chunks a corpus is built from.

    Both :class:`~ragpoison.datagen.loader.Bundle` (read back from disk) and
    :class:`~ragpoison.datagen.generator.GenerationResult` (freshly generated)
    satisfy this, so the validators and the audit share one implementation.
    """

    clean: list[dict]
    distractors: list[dict]
    poisoned: list[dict]
    queries: list[dict]
    corpora: dict[str, dict]


#: Chunk kinds that form the clean part of every corpus.
BASE_KINDS = ("clean", "distractor")

#: How many of the three variants are injected. Note that the variants are
#: numbered 1..3, so this is *not* the set of variant numbers.
INJECTION_COUNTS: tuple[int, ...] = (1, 3)


def base_pool(bundle: CorpusSource) -> list[str]:
    """The clean part of the knowledge base, in a stable order."""
    ids = [row["passage_id"] for row in bundle.clean]
    ids += [row["passage_id"] for row in bundle.distractors]
    return ids


def query_corpus(
    bundle: CorpusSource,
    attack_family: str,
    injection: int,
    query_id: str,
    base: list[str] | None = None,
) -> dict:
    """Knowledge base as seen by one query: base pool plus its own poison."""
    pool = base if base is not None else base_pool(bundle)
    owned = sorted(
        (
            row
            for row in bundle.poisoned
            if row["attack_family"] == attack_family
            and row["query_id"] == query_id
            and int(row["variant"]) <= injection
        ),
        key=lambda row: row["variant"],
    )
    poison_ids = [row["passage_id"] for row in owned]
    return {
        "corpus_id": f"{attack_family}_inj{injection}_{query_id}",
        "attack_family": attack_family,
        "injection": injection,
        "query_id": query_id,
        "chunk_ids": pool + poison_ids,
        "poison_chunk_ids": poison_ids,
        "poison_owner": {chunk_id: query_id for chunk_id in poison_ids},
        "n_chunks": len(pool) + len(poison_ids),
        "design": "query_time",
    }


def iter_query_corpora(
    bundle: CorpusSource, injections: tuple[int, ...] = INJECTION_COUNTS
) -> Iterable[dict]:
    """Yield every query-time corpus exactly once."""
    base = base_pool(bundle)
    families = sorted({row["attack_family"] for row in bundle.poisoned})
    for family in families:
        for injection in injections:
            for query in bundle.queries:
                yield query_corpus(bundle, family, injection, query["query_id"], base=base)


def shared_corpora(bundle: CorpusSource) -> Iterable[dict]:
    """Yield the shared corpora (the corpus-audit view) already on disk."""
    for corpus_id in sorted(bundle.corpora):
        yield bundle.corpora[corpus_id]


def design_document(
    n_clean: int,
    n_distractors: int,
    n_facts: int,
    injections: tuple[int, ...],
    variants: int,
    families: int,
) -> dict:
    """Machine-readable description of both designs, stored in the manifest."""
    base = n_clean + n_distractors
    return {
        "query_time": {
            "description": "base pool plus this query's own poisoned passages",
            "base_pool_chunks": base,
            "n_chunks_per_injection": {str(inj): base + inj for inj in injections},
            "n_corpora": families * len(injections) * n_facts,
            "rationale": (
                "faithful PoisonedRAG setting: a few poisoned chunks inside a larger "
                "clean base; used for every detection and generation metric"
            ),
        },
        "corpus_time": {
            "description": "base pool plus every poisoned passage of one attack family",
            "n_chunks_per_injection": {str(inj): base + n_facts * inj for inj in injections},
            "n_corpora": families * len(injections),
            "files": "corpora/*.json",
            "rationale": (
                "the case the periodic corpus audit exists for: the base really does "
                "contain many poisoned chunks and they must be found on a budget"
            ),
        },
        "variants_per_fact": variants,
    }
