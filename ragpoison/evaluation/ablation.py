"""The offline ablation: fifteen configurations over a frozen signal table.

Decision D12 made this cheap and, more importantly, made it *comparable*: every
configuration scores exactly the same candidates with exactly the same stored signal
values, so a difference between two rows of the summary is a difference in the
fusion rule, not a difference in when or how the signals were measured. Not one API
call happens here.

Protocol:

1. load the joined table for a seed;
2. fit everything that needs fitting - thresholds, weighted alphas, logistic
   coefficients - on the calibration split only;
3. evaluate on the test split only;
4. report the pool-level ``clean_rate`` with a pool-clustered bootstrap interval,
   plus candidate-level precision/recall/F1 and the AUC of the fused score;
5. compare every configuration against the no-defence baseline and against the best
   single-signal configuration with a paired McNemar test over pools.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field

from ..config import Config, load_ablation_matrix, load_config
from ..evaluation.aggregate import (
    baseline_poison_in_topk,
    candidate_metrics,
    family_pools,
    macro_f1,
    pool_index,
    summarise,
)
from ..evaluation.metrics import roc_auc
from ..evaluation.stats import (
    cluster_bootstrap,
    cohens_h,
    mcnemar,
    paired_risk_difference,
    wilcoxon_signed_rank,
)
from ..fusion import calibrate as calibration
from ..fusion.fuse import FusionModel, build_model
from ..fusion.policy import empty_pool_keeps_everything, filter_pool
from ..paths import path as repo_path, rel
from ..signals.load import hard_subset, load_joined, row_key
from ..signals.base import ABLATION_COLUMNS

#: Configurations compared with a paired test. C0 is the floor, C5 was the strongest
#: single signal in the pilot, and C7 is the pair the specification proposes.
REFERENCE_IDS: tuple[str, ...] = ("C0", "C5", "C7")


@dataclass
class AblationRow:
    config_id: str
    label: str
    signals: list[str]
    rule: str
    result: dict = field(default_factory=dict)
    model: FusionModel | None = None
    threshold: calibration.Threshold | None = None
    by_injection: dict[str, dict] = field(default_factory=dict)
    by_family: dict[str, dict] = field(default_factory=dict)


def run_ablation(
    seeds: list[int] | None = None,
    *,
    config: Config | None = None,
    k: int | None = None,
    reps: int = 1000,
    write: bool = True,
    quiet: bool = False,
) -> dict:
    cfg = config or load_config("default.yaml")
    matrix = load_ablation_matrix()
    seed_list = seeds or [42]
    top_k = int(k or cfg.get("retrieval.top_k"))
    records: list[dict] = []
    cell_records: list[dict] = []
    per_seed: dict[int, list[AblationRow]] = {}

    for seed in seed_list:
        rows = load_joined(seed)
        if not rows:
            raise RuntimeError(f"no signal table for seed {seed}: run `python run.py full` first")
        calib = [row for row in rows if row["split"] == "calib"]
        test = [row for row in rows if row["split"] == "test"]
        if not calib or not test:
            raise RuntimeError(f"seed {seed}: calibration or test split is empty")

        pools = pool_index(test)
        baseline = baseline_poison_in_topk(pools, top_k)
        universe = {key for key, count in baseline.items() if count > 0}
        scored_rows = [row for key in universe for row in pools[key]]
        hard = hard_subset(scored_rows)

        for config_id, spec in matrix.items():
            model = build_model(spec.signals, spec.fusion, flip_query_sign=spec.flip_query_sign)
            if spec.fusion == "weighted":
                calibration.fit_alphas(model, calib)
            elif spec.fusion == "logistic":
                calibration.fit_logistic(model, calib)
            threshold = calibration.fit_threshold(model, calib) if spec.fusion != "none" else calibration.Threshold(0.0, 0.0, 0)
            tau = threshold.chosen if spec.fusion != "none" else 0.0

            # Keyed by (corpus_id, query_id, doc_id) - never by doc_id alone.
            scores: dict[tuple, float] = {}
            for row in rows:
                scores[row_key(row)] = model.score(row) if spec.fusion != "none" else 0.0

            filtered = []
            for key in universe:
                subset = pools[key]
                if spec.fusion == "none":
                    filtered.append(empty_pool_keeps_everything(subset, top_k))
                else:
                    filtered.append(filter_pool(subset, scores, tau, top_k))

            candidate = candidate_metrics(scored_rows, scores, tau)
            macro = macro_f1(family_pools(pools), scores, tau)
            auc_value = roc_auc([scores[row_key(row)] for row in scored_rows],
                                [int(row["is_attack_for_query"]) for row in scored_rows])
            auc_hard_value = roc_auc([scores[row_key(row)] for row in hard],
                                     [int(row["is_attack_for_query"]) for row in hard])

            result = summarise(
                filtered,
                universe=universe,
                candidate=candidate,
                macro=macro,
                auc=auc_value,
                auc_hard=auc_hard_value,
                threshold=tau,
                threshold_f1=threshold.f1,
            )
            observed, low, high = cluster_bootstrap(result["_successes"], reps=reps)
            result.update({"clean_rate": observed, "clean_rate_lo": low, "clean_rate_hi": high})

            entry = AblationRow(
                config_id=config_id,
                label=spec.label,
                signals=list(spec.signals),
                rule=spec.fusion,
                result=result,
                model=model,
                threshold=threshold,
            )
            for injection in sorted({int(row["injection"]) for row in scored_rows}):
                subset_keys = {key for key in universe if pools[key][0]["injection"] == injection}
                subset_pools = [pool for pool in filtered if f"{pool.corpus_id}|{pool.query_id}" in subset_keys]
                successes = [int(bool(pool.all_poison_removed)) for pool in subset_pools]
                entry.by_injection[str(injection)] = {
                    "n": len(successes),
                    "clean_rate": sum(successes) / len(successes) if successes else 0.0,
                }
            for family, family_rows in family_pools(pools).items():
                if not family_rows:
                    continue
                family_keys = {
                    f"{pool.corpus_id}|{pool.query_id}"
                    for pool in filtered
                    if pool.attack_family == family
                }
                subset_pools = [pool for pool in filtered if f"{pool.corpus_id}|{pool.query_id}" in family_keys]
                successes = [int(bool(pool.all_poison_removed)) for pool in subset_pools]
                entry.by_family[family] = {
                    "n": len(successes),
                    "clean_rate": sum(successes) / len(successes) if successes else 0.0,
                    "f1": candidate_metrics(family_rows, scores, tau)["f1"],
                }
            # G3: the detection metrics crossed by attack family and injection size.
            # The headline table aggregates them away, and the report's argument is
            # precisely that the two dimensions differ, so the cross has to exist.
            for family in sorted({row["attack_family"] for row in scored_rows}):
                for injection in sorted({int(row["injection"]) for row in scored_rows}):
                    cell_rows = [
                        row
                        for row in scored_rows
                        if row["attack_family"] == family and int(row["injection"]) == injection
                    ]
                    if not cell_rows:
                        continue
                    cell_keys = {f"{row['corpus_id']}|{row['query_id']}" for row in cell_rows}
                    cell_pools = [
                        pool for pool in filtered if f"{pool.corpus_id}|{pool.query_id}" in cell_keys
                    ]
                    successes = [int(bool(pool.all_poison_removed)) for pool in cell_pools]
                    metrics = candidate_metrics(cell_rows, scores, tau)
                    cell_records.append(
                        {
                            "seed": seed,
                            "config": config_id,
                            "attack_family": family,
                            "injection": injection,
                            "n_candidates": len(cell_rows),
                            "n_pools": len(cell_pools),
                            "clean_rate": sum(successes) / len(successes) if successes else 0.0,
                            "precision": round(metrics["precision"], 6),
                            "recall": round(metrics["recall"], 6),
                            "f1": round(metrics["f1"], 6),
                            "fpr": round(metrics["fpr"], 6),
                            "fnr": round(metrics["fnr"], 6),
                            "tp": metrics["tp"],
                            "fp": metrics["fp"],
                            "fn": metrics["fn"],
                            "tn": metrics["tn"],
                        }
                    )
            per_seed.setdefault(seed, []).append(entry)

            row = {
                "seed": seed,
                "config": config_id,
                "label": spec.label,
                "signals": "+".join(spec.signals) or "none",
                "rule": spec.fusion,
                **{key: value for key, value in result.items() if not key.startswith("_")},
            }
            records.append(row)
            if not quiet:
                print(
                    f"  {config_id:<5} clean_rate {row['clean_rate']:.3f} "
                    f"[{row['clean_rate_lo']:.3f},{row['clean_rate_hi']:.3f}]  "
                    f"F1 {row['f1']:.3f}  AUC {_fmt(row['auc'])}",
                    flush=True,
                )

        # Paired tests happen after every configuration is scored.
        by_id = {entry.config_id: entry for entry in per_seed[seed]}
        baseline_successes = by_id["C0"].result["_successes"] if "C0" in by_id else {}
        for entry in per_seed[seed]:
            tests = {}
            for reference_id in REFERENCE_IDS:
                if reference_id not in by_id or reference_id == entry.config_id:
                    continue
                reference = by_id[reference_id].result["_successes"]
                tests[reference_id] = mcnemar(reference, entry.result["_successes"])
            entry.result["tests"] = tests

            # G2: effect size against the undefended baseline, computed from the same
            # discordant pairs the test uses. An odds ratio is deliberately absent:
            # C0 removes poison from exactly zero pools, so McNemar's c cell is always
            # zero and the ratio would be infinite.
            effect = {}
            if baseline_successes and entry.config_id != "C0":
                shared = [key for key in entry.result["_successes"] if key in baseline_successes]
                wins = sum(
                    1
                    for key in shared
                    if entry.result["_successes"][key] == 1 and baseline_successes[key] == 0
                )
                losses = sum(
                    1
                    for key in shared
                    if entry.result["_successes"][key] == 0 and baseline_successes[key] == 1
                )
                n_pairs = len(shared)
                difference, low, high = paired_risk_difference(wins, losses, n_pairs)
                p_candidate = (
                    sum(entry.result["_successes"][key] for key in shared) / n_pairs
                    if n_pairs
                    else 0.0
                )
                p_baseline = (
                    sum(baseline_successes[key] for key in shared) / n_pairs if n_pairs else 0.0
                )
                effect = {
                    "rd": difference,
                    "rd_lo": low,
                    "rd_hi": high,
                    "cohens_h": cohens_h(p_candidate, p_baseline),
                    "wins": wins,
                    "losses": losses,
                    "n_pairs": n_pairs,
                }
            entry.result["effect"] = effect

            for row in records:
                if row["seed"] == seed and row["config"] == entry.config_id:
                    row["mcnemar"] = tests
                    row["rd"] = round(effect.get("rd", 0.0), 6)
                    row["rd_lo"] = round(effect.get("rd_lo", 0.0), 6)
                    row["rd_hi"] = round(effect.get("rd_hi", 0.0), 6)
                    row["cohens_h"] = round(effect.get("cohens_h", 0.0), 6)
                    row["n_pairs"] = effect.get("n_pairs", 0)

    payload = (
        _write(records, per_seed, cell_records, top_k, reps)
        if write
        else {"records": records}
    )
    payload["records"] = records
    payload["cells"] = cell_records
    payload["per_seed"] = per_seed
    return payload


def _mcnemar_cell(test: dict) -> str:
    wins = test["only_candidate"]
    losses = test["only_reference"]
    if wins == 0 and losses == 0:
        return "identical"
    p_value = test["p_value"]
    if p_value < 1e-4:
        shown = "p<1e-4"
    elif p_value >= 0.05:
        shown = f"p={p_value:.2f} n.s."
    else:
        shown = f"p={p_value:.3f}"
    return f"{wins}/{losses} ({shown})"


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


FIELDS: tuple[str, ...] = (
    "seed",
    "config",
    "label",
    "signals",
    "rule",
    "n_pools",
    "n_pools_scored",
    "n_pools_attack_outside_window",
    "clean_rate",
    "clean_rate_lo",
    "clean_rate_hi",
    "atr_before",
    "atr_after",
    "filtered_ratio",
    "refills",
    "abstentions",
    "precision",
    "recall",
    "f1",
    "f1_macro",
    "fpr",
    "fnr",
    "auc",
    "auc_hard",
    "rd",
    "rd_lo",
    "rd_hi",
    "cohens_h",
    "n_pairs",
    "tau",
    "tau_f1",
    "tp",
    "fp",
    "fn",
    "tn",
)


def _write(
    records: list[dict],
    per_seed: dict[int, list[AblationRow]],
    cell_records: list[dict],
    top_k: int,
    reps: int,
) -> dict:
    out_dir = repo_path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary_ablation.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDS))
        writer.writeheader()
        for row in records:
            writer.writerow({key: row.get(key) for key in FIELDS})

    cells_path = out_dir / "summary_ablation_cells.csv"
    cell_fields = (
        "seed",
        "config",
        "attack_family",
        "injection",
        "n_candidates",
        "n_pools",
        "clean_rate",
        "precision",
        "recall",
        "f1",
        "fpr",
        "fnr",
        "tp",
        "fp",
        "fn",
        "tn",
    )
    with cells_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cell_fields))
        writer.writeheader()
        for row in cell_records:
            writer.writerow({key: row.get(key) for key in cell_fields})

    details = []
    for seed, entries in per_seed.items():
        for entry in entries:
            details.append(
                {
                    "seed": seed,
                    "config": entry.config_id,
                    "signals": entry.signals,
                    "rule": entry.rule,
                    "tau_quantile": entry.threshold.quantile if entry.threshold else 0.0,
                    "tau_f1": entry.threshold.f1 if entry.threshold else 0.0,
                    "tau_benign_n": entry.threshold.benign_n if entry.threshold else 0,
                    "alphas": entry.model.alphas if entry.model else {},
                    "coefficients": entry.model.coefficients if entry.model else {},
                    "intercept": entry.model.intercept if entry.model else 0.0,
                    "notes": entry.model.notes if entry.model else [],
                    "by_injection": entry.by_injection,
                    "by_family": entry.by_family,
                    "tests": entry.result.get("tests", {}),
                }
            )
    json_path = out_dir / "summary_ablation_details.json"
    json_path.write_text(json.dumps(details, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

    md_path = repo_path("reports", "ablation.md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(
        _render(records, per_seed, cell_records, top_k, reps), encoding="utf-8", newline="\n"
    )
    return {
        "csv": rel(csv_path),
        "cells": rel(cells_path),
        "details": rel(json_path),
        "report": rel(md_path),
    }


def _render(
    records: list[dict],
    per_seed: dict[int, list[AblationRow]],
    cell_records: list[dict],
    top_k: int,
    reps: int,
) -> str:
    lines: list[str] = []
    lines.append("# Offline ablation - fifteen fusion configurations")
    lines.append("")
    lines.append(
        f"Frozen signal tables, top-k = {top_k}, {reps} bootstrap resamples clustered by pool. "
        "Thresholds, weights and coefficients are fitted on the calibration split; every "
        "reported number is computed on the test split. No API calls."
    )
    lines.append("")

    for seed in sorted(per_seed):
        rows = [row for row in records if row["seed"] == seed]
        lines.append(f"## Seed {seed}")
        lines.append("")
        lines.append(
            "| config | rule | signals | clean_rate | 95% CI | RD vs C0 (pp) | h | poison share: pool → context | filtered | F1 | FPR | AUC | AUC hard |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for row in rows:
            signals = row["signals"].replace("+", "+") if row["signals"] != "none" else "-"
            lines.append(
                f"| {row['config']} | {row['rule']} | {signals} | "
                f"**{row['clean_rate']:.3f}** | "
                f"[{row['clean_rate_lo']:.3f}, {row['clean_rate_hi']:.3f}] | "
                f"{row.get('rd', 0.0) * 100:+.1f} | {row.get('cohens_h', 0.0):.3f} | "
                f"{row['atr_before']:.3f} → {row['atr_after']:.3f} | "
                f"{row['filtered_ratio']:.2f} | {row['f1']:.3f} | {row['fpr']:.3f} | "
                f"{_fmt(row['auc'])} | {_fmt(row['auc_hard'])} |"
            )
        lines.append("")

        baseline = next((row for row in rows if row["config"] == "C0"), None)
        if baseline:
            lines.append(
                f"C0 hands the undefended top-{top_k} straight to the generator, so its clean_rate is "
                f"0.000 by construction: every pool in the denominator is one where the attack did "
                f"reach the window. `poison share: pool -> context` is the share of poisoned "
                f"candidates over all fifteen candidates first, then over the {top_k} that survive: "
                f"it rises even under a good defence whenever the surviving context is small and "
                f"still contains one poisoned passage."
            )
            lines.append("")

        lines.append("### Paired comparison (McNemar, pools where two configurations disagree)")
        lines.append("")
        lines.append("| config | vs C0 | vs C5 | vs C7 |")
        lines.append("|---|---|---|---|")
        for row in rows:
            tests = row.get("mcnemar") or {}
            cells = []
            for reference in REFERENCE_IDS:
                if reference == row["config"]:
                    cells.append("-")
                elif reference in tests:
                    cells.append(_mcnemar_cell(tests[reference]))
                else:
                    cells.append("-")
            lines.append(f"| {row['config']} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append(
            "Each cell is `wins/losses (p)`: pools where this configuration removed all the poison "
            "and the reference did not, over the reverse. A blank result means the two never disagree."
        )
        lines.append("")

        lines.append("### Effect size against no defence")
        lines.append("")
        lines.append("| config | wins | losses | fixed pools | risk difference | 95% CI | Cohen's h |")
        lines.append("|---|---|---|---|---|---|---|")
        for entry in per_seed[seed]:
            effect = entry.result.get("effect") or {}
            if not effect:
                continue
            lines.append(
                f"| {entry.config_id} | {effect['wins']} | {effect['losses']} | "
                f"{effect['n_pairs']} | {effect['rd'] * 100:+.1f} pp | "
                f"[{effect['rd_lo'] * 100:+.1f}, {effect['rd_hi'] * 100:+.1f}] | "
                f"{effect['cohens_h']:.3f} |"
            )
        lines.append("")
        lines.append(
            "The risk difference is the primary measure: same unit as clean_rate, same discordant "
            "pairs as the McNemar test above. Cohen's h is a standardised secondary measure computed "
            "on the marginal proportions, so it ignores the pairing and usually understates the "
            "effect. No odds ratio is reported: no-defence removes poison from exactly zero pools, "
            "so the discordant cell that would divide it is always zero."
        )
        lines.append("")

        lines.append("### Cell-level contrasts (Wilcoxon signed-rank, n = 12)")
        lines.append("")
        by_config: dict[str, dict[tuple, float]] = {}
        for cell in cell_records:
            if cell["seed"] != seed:
                continue
            by_config.setdefault(cell["config"], {})[
                (cell["attack_family"], cell["injection"])
            ] = cell["f1"]
        lines.append("| contrast (F1 per cell) | n | W+ | p (exact) | median difference |")
        lines.append("|---|---|---|---|---|")
        for first, second in (("C8l", "C4"), ("C8", "C4"), ("C8l", "C8"), ("C7", "C4")):
            left = by_config.get(first, {})
            right = by_config.get(second, {})
            shared_keys = [key for key in left if key in right]
            if not shared_keys:
                continue
            result = wilcoxon_signed_rank([(left[key], right[key]) for key in shared_keys])
            ordered = sorted(left[key] - right[key] for key in shared_keys)
            middle = len(ordered) // 2
            median = (
                ordered[middle]
                if len(ordered) % 2
                else (ordered[middle - 1] + ordered[middle]) / 2
            )
            lines.append(
                f"| {first} vs {second} | {result['n']} | {result['w_plus']:.0f} | "
                f"{result['p_value']:.4f} | {median:+.3f} |"
            )
        lines.append("")
        lines.append(
            "n = 12 cells (6 attacks x 2 injection sizes), every cell this seed has. At that size "
            "the smallest attainable two-sided p-value is about 0.0005, so significance is cheap and "
            "the direction of the median difference carries more information than the p-value."
        )
        lines.append("")

        lines.append("### Detection metrics per attack and injection size")
        lines.append("")
        cell_keys = sorted(
            {
                (cell["attack_family"], cell["injection"])
                for cell in cell_records
                if cell["seed"] == seed
            }
        )
        for metric in ("clean_rate", "precision", "recall", "f1", "fpr"):
            table = {
                (cell["config"], cell["attack_family"], cell["injection"]): cell[metric]
                for cell in cell_records
                if cell["seed"] == seed
            }
            lines.append(f"**{metric}**")
            lines.append("")
            header = "| config | " + " | ".join(
                f"{family.split('_')[0]}\u00b7inj{injection}" for family, injection in cell_keys
            ) + " |"
            lines.append(header)
            lines.append("|---|" + "---|" * len(cell_keys))
            for row in rows:
                values = [
                    f"{table.get((row['config'], family, injection), 0.0):.2f}"
                    for family, injection in cell_keys
                ]
                lines.append(f"| {row['config']} | " + " | ".join(values) + " |")
            lines.append("")

        entries = per_seed[seed]
        lines.append("### Clean rate by injection size")
        lines.append("")
        injections = sorted({key for entry in entries for key in entry.by_injection})
        lines.append("| config | " + " | ".join(f"inj={value}" for value in injections) + " |")
        lines.append("|---|" + "---|" * len(injections))
        for entry in entries:
            cells = [
                f"{entry.by_injection.get(value, {}).get('clean_rate', 0.0):.3f} (n={entry.by_injection.get(value, {}).get('n', 0)})"
                for value in injections
            ]
            lines.append(f"| {entry.config_id} | " + " | ".join(cells) + " |")
        lines.append("")

        lines.append("### Clean rate by attack family")
        lines.append("")
        families = sorted({key for entry in entries for key in entry.by_family})
        short = [family.split("_")[0] for family in families]
        lines.append("| config | " + " | ".join(short) + " |")
        lines.append("|---|" + "---|" * len(families))
        for entry in entries:
            cells = [
                f"{entry.by_family.get(family, {}).get('clean_rate', 0.0):.2f}" for family in families
            ]
            lines.append(f"| {entry.config_id} | " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("## How to read this table")
    lines.append("")
    lines.append(
        "- `clean_rate` is the fraction of pools whose poisoned passages are all gone from the "
        f"top-{top_k} that would be handed to the generator. It is measured only on pools where "
        "the attack reached the undefended window, and the excluded count is in the CSV."
    )
    lines.append(
        "- `refills` counts pools where a defence had to put a filtered passage back to reach "
        f"k={top_k}. A defence that refills is not defending that pool."
    )
    lines.append(
        "- AUC is computed on the fused score, not on a single signal, and the hard subset "
        "restricts it to poison against the clean passages that actually answered the question."
    )
    lines.append(
        "- The McNemar p-values use exact binomial tails over the discordant pools; with a few "
        "hundred pools, p < 0.05 needs a clear margin, which is the intent."
    )
    lines.append(
        "- **What clean_rate cannot see.** It asks only whether the poisoned passages left the "
        "context, so a defence that also removes the genuine evidence scores well here while "
        "destroying the answer. C1 is the case to watch: it reaches a clean_rate near 0.5 because "
        "it cuts the top-ranked candidates, where the poison usually sits, and it is just as happy "
        "to cut the real evidence. Answerability after filtering needs a generation from the "
        "filtered context, which this offline stage deliberately does not do."
    )
    lines.append("")
    return "\n".join(lines)
