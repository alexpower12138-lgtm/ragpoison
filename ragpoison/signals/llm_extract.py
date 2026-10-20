"""LLM-backed signal extraction: S3, S3b, S4, S5.

This is the only stage that costs money, so the shape of the work matters:

* every call is built first for **all** selected instances, then submitted as one
  batch. That gives maximum throughput and one progress line;
* S4 depends only on the passage text, so identical texts are issued once per
  process and paid once ever (the on-disk cache handles later runs);
* S5 batches several passages per request, and reuses S4 for its instruction view
  when S4 is enabled, so the view is never paid for twice;
* everything derivable without a model (densities, majority vote, the mean of the
  views) is computed afterwards from the parsed answers.

The output is a second table, joined to the local one in WP6 on
``(corpus_id, query_id, doc_id)``. Keeping them apart means the local table can be
rebuilt for free at any time and this expensive one is never touched.
"""

from __future__ import annotations

import json

from ..config import Config, load_config
from ..datagen.corpora import iter_query_corpora
from ..datagen.loader import load_bundle
from ..llm import mock as mock_responders
from ..llm import prompts
from ..llm.client import Call, LLMClient
from ..llm.parse import parse_answer, parse_flags, parse_score
from ..paths import path as repo_path
from . import s3_freq_density, s3b_majority
from .extract import _label, build_context

LLM_SIGNALS: tuple[str, ...] = ("s3", "s3b", "s4", "s5")
VIEWS: tuple[str, ...] = ("relevance", "instruction", "contradiction", "inducement")
#: Fallback reference density when the calibration split is not in the run.
DEFAULT_C_REF = 0.10


def _views_enabled(config: Config) -> dict[str, bool]:
    return {
        "relevance": bool(config.get("detectors.multiview.enable_relevance")),
        "instruction": bool(config.get("detectors.multiview.enable_instruction")),
        "contradiction": bool(config.get("detectors.multiview.enable_contradiction")),
        "inducement": bool(config.get("detectors.multiview.enable_inducement")),
    }


def _batch(items: list, size: int) -> list[list]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def extract_llm(
    seed: int = 42,
    *,
    family: str | None = None,
    injection: int | None = None,
    query_id: str | None = None,
    limit: int | None = None,
    config: Config | None = None,
    top_n: int | None = None,
    mock: bool = False,
    concurrency: int | None = None,
    budget_cny: float | None = None,
    write: bool = True,
    quiet: bool = False,
) -> dict:
    cfg = config or load_config("default.yaml")
    pool = int(top_n or cfg.get("retrieval.top_n"))
    batch_size = int(cfg.get("detectors.mv_batch_size"))
    max_tokens = int(cfg.get("llm.max_tokens_json"))
    bundle = load_bundle(seed)
    views = _views_enabled(cfg)
    s4_enabled = bool(cfg.get("detectors.enable_instruction_judge"))
    s5_enabled = bool(cfg.get("detectors.enable_multiview_judge"))

    selected = []
    for corpus in iter_query_corpora(bundle):
        if family and corpus["attack_family"] != family:
            continue
        if injection is not None and int(corpus["injection"]) != injection:
            continue
        if query_id and corpus["query_id"] != query_id:
            continue
        if limit is not None and len(selected) >= limit:
            break
        selected.append(corpus)

    # ---------------------------------------------------------------- planning
    plans: list[dict] = []
    calls: list[Call] = []
    s4_seen: dict[str, Call] = {}

    for corpus in selected:
        query = next(row for row in bundle.queries if row["query_id"] == corpus["query_id"])
        context, _bm25, _tfidf = build_context(bundle, corpus, query, {}, cfg, pool)
        ids = context.ids()
        plan = {"corpus": corpus, "query": query, "context": context, "ids": ids, "calls": {}}

        for candidate in context.candidates:
            call = Call(
                system=prompts.S3_SYSTEM,
                user=prompts.S3_USER.format(passage=candidate.text, question=query["question"]),
                stage="s3_answer",
                max_tokens=max_tokens,
                extra={"corpus_id": corpus["corpus_id"], "doc_id": candidate.doc_id},
            )
            plan["calls"].setdefault("s3", []).append((candidate.doc_id, call))
            calls.append(call)

            if s4_enabled:
                s4_call = Call(
                    system=prompts.S4_SYSTEM,
                    user=prompts.S4_USER.format(passage=candidate.text),
                    stage="s4_instruction",
                    max_tokens=max_tokens,
                    extra={"doc_id": candidate.doc_id},
                )
                # Identical text is judged once per process, not once per instance.
                if candidate.doc_id not in s4_seen:
                    s4_seen[candidate.doc_id] = s4_call
                    calls.append(s4_call)
                plan["calls"].setdefault("s4", []).append((candidate.doc_id, s4_call))

        if s5_enabled:
            for view in VIEWS:
                if not views.get(view, False):
                    continue
                if view == "instruction" and s4_enabled:
                    continue  # reused from S4, never paid twice
                for batch_index, chunk in enumerate(_batch(context.candidates, batch_size)):
                    passages = "\n".join(
                        f"[{position}] {candidate.text}"
                        for position, candidate in enumerate(chunk, start=1)
                    )
                    call = Call(
                        system=prompts.S5_SYSTEM,
                        user=prompts.S5_USER.format(
                            question=query["question"],
                            passages=passages,
                            view_description=prompts.S5_VIEWS[view],
                        ),
                        stage=f"s5_{view}",
                        max_tokens=max_tokens * 3,
                        extra={
                            "corpus_id": corpus["corpus_id"],
                            "view": view,
                            "ids": [candidate.doc_id for candidate in chunk],
                        },
                    )
                    plan["calls"].setdefault("s5", []).append((chunk, call))
                    calls.append(call)

        plans.append(plan)

    # ---------------------------------------------------------------- execution
    client = LLMClient(
        cfg,
        concurrency=concurrency,
        budget_cny=budget_cny,
        mock=mock,
        mock_responder=mock_responders.signal_responder,
        quiet=quiet,
    )
    if not quiet:
        print(
            f"llm signals: {len(plans)} instances, {len(calls)} requests "
            f"({len(s4_seen)} distinct passages for S4) ...",
            flush=True,
        )
    results = client.complete_many(calls, label="signals    ", progress_every=200)
    outcome_by_call = {id(call): result for call, result in zip(calls, results)}

    # ---------------------------------------------------------------- assembly
    s4_cache: dict[str, tuple[float, str, bool]] = {}
    rows: list[dict] = []
    parse_stats = {"s3_fail": 0, "s4_fail": 0, "s5_fail": 0, "s3_unknown": 0, "s3_total": 0}

    for plan in plans:
        context = plan["context"]
        corpus = plan["corpus"]
        query = plan["query"]
        answers: dict[str, str] = {}
        answer_ok: dict[str, bool] = {}

        for doc_id, call in plan["calls"].get("s3", []):
            result = outcome_by_call[id(call)]
            parsed = parse_answer(result.text) if result.ok else None
            parse_stats["s3_total"] += 1
            if parsed is None:
                parse_stats["s3_fail"] += 1
                answers[doc_id] = "UNKNOWN"
                answer_ok[doc_id] = False
            else:
                answers[doc_id] = parsed[0]
                answer_ok[doc_id] = True
                if parsed[0].strip().upper().startswith("UNKNOWN"):
                    parse_stats["s3_unknown"] += 1

        for doc_id, call in plan["calls"].get("s4", []):
            if doc_id in s4_cache:
                continue
            result = outcome_by_call[id(call)]
            parsed = parse_score(result.text) if result.ok else None
            if parsed is None:
                parse_stats["s4_fail"] += 1
                s4_cache[doc_id] = (0.0, "parse_fail", False)
            else:
                s4_cache[doc_id] = (parsed[0], parsed[1], True)

        view_flags: dict[str, dict[str, float]] = {view: {} for view in VIEWS}
        for chunk, call in plan["calls"].get("s5", []):
            result = outcome_by_call[id(call)]
            view = call.extra["view"]
            expected = list(range(1, len(chunk) + 1))
            parsed = parse_flags(result.text, expected) if result.ok else None
            if parsed is None:
                parse_stats["s5_fail"] += 1
                for candidate in chunk:
                    view_flags[view][candidate.doc_id] = 0.0
                continue
            if len(parsed) < len(chunk):
                parse_stats["s5_fail"] += 1
            for position, candidate in enumerate(chunk, start=1):
                view_flags[view][candidate.doc_id] = float(parsed.get(position, 0.0))

        majority = s3b_majority.scores(answers)

        for candidate in context.candidates:
            answer = answers.get(candidate.doc_id, "UNKNOWN")
            c_density = s3_freq_density.density(candidate.text, query["question"], answer)
            s4_score, s4_category, s4_ok = s4_cache.get(
                candidate.doc_id, (0.0, "disabled", True)
            )
            if s4_enabled:
                # The instruction view is the S4 verdict, so it is never asked twice.
                view_flags["instruction"][candidate.doc_id] = s4_score
            view_values = [
                view_flags.get(view, {}).get(candidate.doc_id, 0.0)
                for view in VIEWS
                if views.get(view, False) or view == "instruction"
            ]
            s5_score = sum(view_values) / len(view_values) if view_values else 0.0

            label_class, is_attack = _label(candidate.doc_id, corpus, bundle)
            rows.append(
                {
                    "seed": seed,
                    "attack_family": corpus["attack_family"],
                    "injection": int(corpus["injection"]),
                    "corpus_id": corpus["corpus_id"],
                    "query_id": query["query_id"],
                    "fact_id": query["fact_id"],
                    "split": query["split"],
                    "doc_id": candidate.doc_id,
                    "rank": candidate.rank,
                    "answer": answer,
                    "answer_parsed": int(answer_ok.get(candidate.doc_id, False)),
                    "s3_c": round(c_density, 6),
                    "s3b": round(majority.get(candidate.doc_id, 0.0), 6),
                    "s4": round(s4_score, 6),
                    "s4_category": s4_category,
                    "s4_parsed": int(s4_ok),
                    "s5": round(s5_score, 6),
                    **{
                        f"s5_{view}": round(view_flags.get(view, {}).get(candidate.doc_id, 0.0), 6)
                        for view in VIEWS
                    },
                    "label_class": label_class,
                    "is_attack_for_query": is_attack,
                }
            )

    # --------------------------------------------------- density normalisation
    calib_densities = [
        row["s3_c"] for row in rows if row["split"] == "calib" and row["label_class"] == "clean"
    ]
    if len(calib_densities) >= 30:
        c_ref, c_ref_source = _median(calib_densities), "calib_clean"
    else:
        all_clean = [row["s3_c"] for row in rows if row["label_class"] == "clean"]
        c_ref = _median(all_clean) if all_clean else DEFAULT_C_REF
        c_ref_source = "all_clean_fallback"
    if c_ref <= 0:
        c_ref, c_ref_source = DEFAULT_C_REF, "default"

    for row in rows:
        row["s3"] = round(
            s3_freq_density.score_from_density(row["s3_c"], row["answer"], c_ref), 6
        )

    # ---------------------------------------------------------------- persistence
    written: list[str] = []
    if write and rows:
        target_dir = repo_path("results", "signals", f"seed{seed}")
        target_dir.mkdir(parents=True, exist_ok=True)
        by_corpus: dict[str, list[dict]] = {}
        for row in rows:
            key = f"{row['attack_family']}_inj{row['injection']}"
            by_corpus.setdefault(key, []).append(row)
        for key, payload in sorted(by_corpus.items()):
            path = target_dir / f"{key}.llm.jsonl"
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                for row in payload:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written.append(path.relative_to(repo_path()).as_posix())

    snapshot = prompts.snapshot(
        "signals_llm",
        f"{prompts.S3_VERSION}+{prompts.S4_VERSION}+{prompts.S5_VERSION}",
        {
            "s3_system": prompts.S3_SYSTEM,
            "s3_user": prompts.S3_USER,
            "s4_system": prompts.S4_SYSTEM,
            "s4_user": prompts.S4_USER,
            "s5_system": prompts.S5_SYSTEM,
            "s5_user": prompts.S5_USER,
            "s5_views": prompts.S5_VIEWS,
        },
    )
    client.write_spend_line("signals_llm")

    total_calls = parse_stats["s3_total"] or 1
    return {
        "seed": seed,
        "mock": mock,
        "instances": len(plans),
        "rows": len(rows),
        "requests": len(calls),
        "distinct_s4_passages": len(s4_seen),
        "c_ref": c_ref,
        "c_ref_source": c_ref_source,
        "parse": parse_stats,
        "s3_parse_fail_rate": parse_stats["s3_fail"] / total_calls,
        "s3_unknown_rate": parse_stats["s3_unknown"] / total_calls,
        "s4_parse_fail_rate": (
            parse_stats["s4_fail"] / len(s4_seen) if s4_seen else 0.0
        ),
        "s5_batches": sum(len(plan["calls"].get("s5", [])) for plan in plans),
        "llm": client.summary(),
        "files": written,
        "snapshot": snapshot.relative_to(repo_path()).as_posix(),
        "rows_data": rows,
    }


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[len(ordered) // 2])
