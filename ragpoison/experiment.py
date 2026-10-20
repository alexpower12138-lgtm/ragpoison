"""Experiment runner: one command per stage of the experiment matrix.

A "run" is the full pipeline for a set of instances: build the local signal table,
build the LLM signal table, join them, and report what came out. The three
configured modes differ only in how many instances they cover:

* ``pilot``      R1, about 120 instances, the acceptance gate before real spending
* ``full``       R2, every instance on seed 42
* ``replicate``  R3, every instance on seed 43

Nothing here recomputes a signal: the extractors own that, and every call they make
is cached, so re-running a mode resumes instead of paying twice.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .config import Config, load_config
from .evaluation.metrics import mean, roc_auc
from .paths import path as repo_path, rel
from .signals.base import ABLATION_COLUMNS
from .signals.extract import LOCAL_SIGNALS, extract
from .signals.llm_extract import LLM_SIGNALS, extract_llm
from .datagen.loader import load_bundle

MODES: tuple[str, ...] = ("pilot", "full", "replicate")

#: Directory naming. The pilot writes beside the full table rather than over it.
def signals_dir(mode: str, seed: int) -> Path:
    suffix = "_pilot" if mode == "pilot" else ""
    return repo_path("results", "signals", f"seed{seed}{suffix}")


def _query_budget(mode: str, cfg: Config) -> int | None:
    if mode == "pilot":
        count = int(cfg.get("experiment.pilot_queries"))
        return count or None
    if mode == "replicate":
        count = int(cfg.get("experiment.replicate_queries"))
        return count or None
    return None


def run(
    mode: str,
    *,
    seeds: list[int] | None = None,
    limit: int | None = None,
    mock: bool = False,
    dry_run: bool = False,
    quiet: bool = False,
) -> dict:
    """Run one experiment mode and write its acceptance report."""
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    cfg = load_config(f"{mode}.yaml")
    seed_list = seeds or [int(seed) for seed in cfg.get("experiment.seeds")]
    injections = [int(value) for value in cfg.get("experiment.injection_counts")]
    queries_per_seed = _query_budget(mode, cfg)
    top_n = int(cfg.get("retrieval.top_n"))
    concurrency = int(cfg.get("runtime.concurrency"))
    budget = float(cfg.get("runtime.budget_cny"))

    results: list[dict] = []
    for seed in seed_list:
        bundle = load_bundle(seed)
        selected = bundle.queries[:queries_per_seed] if queries_per_seed else bundle.queries
        query_ids = [row["query_id"] for row in selected]
        target = signals_dir(mode, seed)

        # The extractors each build every instance they are given, so the injection
        # filter has to be pushed down into them. Filtering afterwards would pay for
        # instances the configuration does not ask for: the pilot asks for injection
        # size 1 only, and extracting both sizes would have doubled its cost.
        injection_filter = injections[0] if len(injections) == 1 else None
        families = cfg.get("experiment.attacks") or []
        expected = len(query_ids) * len(families) * len(injections)

        if not quiet:
            print(
                f"[{mode}] seed {seed}: {len(query_ids)} queries x {len(families)} families x "
                f"{injections} = {expected} instances, signals -> {rel(target)}",
                flush=True,
            )

        local = extract(
            seed=seed,
            query_ids=query_ids,
            injection=injection_filter,
            limit=limit,
            config=cfg,
            top_n=top_n,
            out_dir=target,
        )
        llm = extract_llm(
            seed=seed,
            query_ids=query_ids,
            injection=injection_filter,
            limit=limit,
            config=cfg,
            top_n=top_n,
            mock=mock,
            concurrency=concurrency,
            budget_cny=budget,
            out_dir=target,
            dry_run=dry_run,
            quiet=quiet,
        )

        if dry_run:
            if not quiet:
                print(
                    f"  requests {llm['requests']}, unique prompts {llm['unique_requests']}, "
                    f"already cached {llm['already_cached']}, to pay {llm['to_pay']} "
                    f"(~{llm['estimated_cost_cny']:.2f} CNY at {cfg.get('llm.price_input_cny_per_mtok')}/Mtok)"
                )
            results.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "queries": len(query_ids),
                    "dry_run": True,
                    "instances": llm["instances"],
                    "requests": llm["requests"],
                    "unique_requests": llm["unique_requests"],
                    "already_cached": llm["already_cached"],
                    "to_pay": llm["to_pay"],
                    "estimated_cost_cny": llm["estimated_cost_cny"],
                }
            )
            continue

        rows = _join(target)
        rows = [row for row in rows if int(row["injection"]) in injections]
        summary = _summarise(rows, seed, mode)
        summary.update(
            {
                "mode": mode,
                "seed": seed,
                "queries": len(query_ids),
                "expected_instances": expected,
                "local_rows": local["rows"],
                "llm_rows": llm["rows"],
                "requests": llm["requests"],
                "mock": mock,
                "parse": llm["parse"],
                "failed_calls": llm.get("failed_calls", 0),
                "failed_batches": llm.get("failed_batches", 0),
                "s3_parse_fail_rate": llm["s3_parse_fail_rate"],
                "s4_parse_fail_rate": llm["s4_parse_fail_rate"],
                "s3_unknown_rate": llm["s3_unknown_rate"],
                "c_ref": llm["c_ref"],
                "c_ref_source": llm["c_ref_source"],
                "llm": llm["llm"],
                "signals_dir": rel(target),
            }
        )
        results.append(summary)

        md_path = repo_path("reports", f"run_{mode}_seed{seed}.md")
        json_path = repo_path("results", f"run_{mode}_seed{seed}.json")
        md_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps({k: v for k, v in summary.items() if k != "samples"}, indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        md_path.write_text(_render(summary), encoding="utf-8", newline="\n")
        summary["_report"] = rel(md_path)

        if not quiet:
            print(_console(summary))

    return {"mode": mode, "runs": results}


# --------------------------------------------------------------------------- #
# Join and metrics
# --------------------------------------------------------------------------- #


def _cells(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _join(directory: Path) -> list[dict]:
    """Join the local and LLM tables on (corpus_id, query_id, doc_id)."""
    local: dict[tuple, dict] = {}
    llm: dict[tuple, dict] = {}
    for path in sorted(directory.glob("*.jsonl")):
        target = llm if ".llm." in path.name else local
        for row in _cells(path):
            target[(row["corpus_id"], row["query_id"], row["doc_id"])] = row

    joined: list[dict] = []
    for key, row in local.items():
        merged = dict(row)
        other = llm.get(key)
        if other is not None:
            for name in LLM_SIGNALS:
                merged[name] = other.get(name, 0.0)
            merged["s3c"] = other.get("s3_c", 0.0)
            merged["answer"] = other.get("answer", "")
            merged["s3_unknown"] = other.get("s3_unknown", 1)
            merged["s4_category"] = other.get("s4_category", "")
        else:
            for name in LLM_SIGNALS:
                merged[name] = 0.0
            merged["s3c"] = 0.0
            merged["answer"] = ""
            merged["s3_unknown"] = 1
            merged["s4_category"] = ""
        joined.append(merged)
    return joined


SIGNAL_TABLES: dict[str, str] = {
    name: table for name, (table, _column) in ABLATION_COLUMNS.items()
}


def _summarise(rows: list[dict], seed: int, mode: str) -> dict:
    columns = list(ABLATION_COLUMNS)
    labels = [int(row["is_attack_for_query"]) for row in rows]
    auc: dict[str, float | None] = {}
    for name in columns:
        _table, column = ABLATION_COLUMNS[name]
        scores = [float(row.get(column, 0.0)) for row in rows]
        auc[name] = roc_auc(scores, labels)

    # Hard subset: drop the negatives that simply could not answer the question.
    # They are easy to separate and they flatter content-level signals. What is
    # left is "poison against the genuine evidence", which is the comparison the
    # project actually cares about.
    hard = [
        row
        for row in rows
        if int(row["is_attack_for_query"]) == 1
        or (row["label_class"] == "clean" and not int(row.get("s3_unknown", 1)))
    ]
    hard_labels = [int(row["is_attack_for_query"]) for row in hard]
    auc_hard: dict[str, float | None] = {}
    for name in columns:
        column = ABLATION_COLUMNS[name][1]
        auc_hard[name] = roc_auc([float(row.get(column, 0.0)) for row in hard], hard_labels)

    per_label: dict[str, dict[str, float]] = {}
    by_label: dict[str, list[dict]] = {}
    for row in rows:
        by_label.setdefault(row["label_class"], []).append(row)
    for label_class, subset in sorted(by_label.items()):
        entry: dict[str, float] = {"n": float(len(subset))}
        for name in columns:
            _table, column = ABLATION_COLUMNS[name]
            entry[name] = mean([float(row.get(column, 0.0)) for row in subset])
        per_label[label_class] = entry

    per_family: dict[str, dict[str, float]] = {}
    for family in sorted({row["attack_family"] for row in rows}):
        subset = [row for row in rows if row["attack_family"] == family]
        family_labels = [int(row["is_attack_for_query"]) for row in subset]
        per_family[family] = {
            name: roc_auc(
                [float(row.get(ABLATION_COLUMNS[name][1], 0.0)) for row in subset],
                family_labels,
            )
            for name in columns
        }

    samples = _samples(rows)

    return {
        "n_rows": len(rows),
        "n_instances": len({(row["corpus_id"], row["query_id"]) for row in rows}),
        "n_positives": sum(labels),
        "auc": auc,
        "auc_hard": auc_hard,
        "n_hard": len(hard),
        "per_label": per_label,
        "per_family_auc": per_family,
        "samples": samples,
    }


def _samples(rows: list[dict], count: int = 3, top: int = 5) -> list[dict]:
    """A few whole instances, for a human to eyeball."""
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["corpus_id"], row["query_id"]), []).append(row)
    out: list[dict] = []
    for (corpus_id, query_id), subset in list(sorted(grouped.items()))[:count]:
        subset.sort(key=lambda row: row["rank"])
        out.append(
            {
                "corpus_id": corpus_id,
                "query_id": query_id,
                "rows": [
                    {
                        "rank": row["rank"],
                        "doc_id": row["doc_id"],
                        "label": row["label_class"],
                        **{
                            name: round(float(row.get(ABLATION_COLUMNS[name][1], 0.0)), 3)
                            for name in ABLATION_COLUMNS
                        },
                        "answer": str(row.get("answer", ""))[:20],
                    }
                    for row in subset[:top]
                ],
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _fmt(value: float | None) -> str:
    if value is None:
        return "  n/a"
    return f"{value:.3f}"


def _render(summary: dict) -> str:
    lines: list[str] = []
    lines.append(f"# Run report - {summary['mode']} - seed {summary['seed']}")
    lines.append("")
    if summary["mock"]:
        lines.append("> **MOCK RUN.** No model was called. Numbers are a plumbing test only.")
        lines.append("")
    lines.append(f"- queries: {summary['queries']}")
    lines.append(
        f"- instances: {summary['n_instances']} (configured: {summary['expected_instances']}), "
        f"rows: {summary['n_rows']}"
    )
    lines.append(f"- positives (targeted poisoned candidates): {summary['n_positives']}")
    lines.append(f"- signal tables: `{summary['signals_dir']}`")
    lines.append(f"- requests issued: {summary['requests']}")
    parse = summary["parse"]
    lines.append(
        f"- parse failures: S3 {parse['s3_fail']}/{parse['s3_total']} "
        f"({summary['s3_parse_fail_rate']:.1%}), S4 {parse['s4_fail']} "
        f"({summary['s4_parse_fail_rate']:.1%}), S5 batches {parse['s5_fail']}"
    )
    if summary.get("failed_calls"):
        lines.append(
            f"- **WARNING: {summary['failed_calls']} call(s) still failed after retries.** "
            "Their rows are marked `call_failed`; re-run the same command to heal them "
            "(already-cached calls are not paid twice)."
        )
    lines.append(
        f"- S3 UNKNOWN answers: {summary['s3_unknown_rate']:.1%}  "
        f"(c_ref={summary['c_ref']:.4f}, source {summary['c_ref_source']})"
    )
    llm = summary["llm"]
    lines.append(
        f"- api calls {llm['calls']}, cache hits {llm['cache_hits']}, failures {llm['failures']}, "
        f"tokens {llm['prompt_tokens']} in / {llm['completion_tokens']} out"
    )
    lines.append("")
    lines.append("## Per-signal AUC (targeted poison vs everything else)")
    lines.append("")
    lines.append("| signal | table | AUC |")
    lines.append("|---|---|---|")
    for name in ABLATION_COLUMNS:
        lines.append(f"| {name} | {ABLATION_COLUMNS[name][0]} | {_fmt(summary['auc'][name])} |")
    lines.append("")
    lines.append(
        f"### Hard subset: poison against the genuine evidence only "
        f"({summary['n_hard']} candidates, the negatives that did answer the question)"
    )
    lines.append("")
    lines.append("| signal | AUC |")
    lines.append("|---|---|")
    for name in ABLATION_COLUMNS:
        lines.append(f"| {name} | {_fmt(summary['auc_hard'][name])} |")
    lines.append("")
    lines.append("## Mean score by label")
    lines.append("")
    header = "| label | n |" + "".join(f" {name} |" for name in ABLATION_COLUMNS)
    lines.append(header)
    lines.append("|---|---|" + "---|" * len(ABLATION_COLUMNS))
    for label_class, entry in summary["per_label"].items():
        row = f"| {label_class} | {int(entry['n'])} |"
        row += "".join(f" {entry[name]:.3f} |" for name in ABLATION_COLUMNS)
        lines.append(row)
    lines.append("")
    lines.append("## Per-family AUC")
    lines.append("")
    lines.append("| family |" + "".join(f" {name} |" for name in ABLATION_COLUMNS))
    lines.append("|---|" + "---|" * len(ABLATION_COLUMNS))
    for family, entry in summary["per_family_auc"].items():
        row = f"| {family} |"
        row += "".join(f" {_fmt(entry[name])} |" for name in ABLATION_COLUMNS)
        lines.append(row)
    lines.append("")
    lines.append("## Sample instances (top 5 candidates each)")
    lines.append("")
    for sample in summary["samples"]:
        lines.append(f"### {sample['corpus_id']} / {sample['query_id']}")
        lines.append("")
        header = "| rk | doc | label | answer |" + "".join(f" {name} |" for name in ABLATION_COLUMNS)
        lines.append(header)
        lines.append("|---|---|---|---|" + "---|" * len(ABLATION_COLUMNS))
        for row in sample["rows"]:
            line = f"| {row['rank']} | {row['doc_id']} | {row['label']} | {row['answer']} |"
            line += "".join(f" {row[name]:.3f} |" for name in ABLATION_COLUMNS)
            lines.append(line)
        lines.append("")
    lines.append("## Reading this")
    lines.append("")
    lines.append(
        "- AUC is computed over candidates, with the targeted poisoned passage as the only "
        "positive class. 0.5 is chance; a signal that flags 'the passages that answer the "
        "question' also scores well here, so check the per-label means and the samples before "
        "concluding that a signal detects poison."
    )
    lines.append(
        "- A parse-failure rate above 5% or an S3 UNKNOWN rate near 100% means the prompt needs "
        "work before the full run, because changing a prompt invalidates the cache."
    )
    lines.append("")
    return "\n".join(lines)


def _console(summary: dict) -> str:
    lines = [
        "",
        f"=== {summary['mode']} seed {summary['seed']}: "
        f"{summary['n_instances']} instances, {summary['n_rows']} rows, "
        f"{summary['n_positives']} positives",
        f"parse failures: S3 {summary['s3_parse_fail_rate']:.1%}, "
        f"S4 {summary['s4_parse_fail_rate']:.1%}, S5 {summary['parse']['s5_fail']} batches",
        f"S3 UNKNOWN {summary['s3_unknown_rate']:.1%}   c_ref {summary['c_ref']:.4f} "
        f"({summary['c_ref_source']})",
        "",
        f"{'signal':<20}{'table':>7}{'AUC':>8}{'AUC hard':>10}",
    ]
    for name in ABLATION_COLUMNS:
        lines.append(
            f"{name:<20}{ABLATION_COLUMNS[name][0]:>7}{_fmt(summary['auc'][name]):>8}"
            f"{_fmt(summary['auc_hard'][name]):>10}"
        )
    llm = summary["llm"]
    lines.append("")
    lines.append(
        f"calls {llm['calls']}, cache hits {llm['cache_hits']}, failures {llm['failures']}, "
        f"tokens {llm['prompt_tokens']} in / {llm['completion_tokens']} out"
    )
    lines.append(f"report: {summary.get('_report', '')}")
    return "\n".join(lines)
