"""End-to-end defence: generate an answer from the *filtered* context.

The offline ablation measures whether the poison leaves the context. That is not the
same question as whether the attack stops working, because a filter can also remove
the genuine evidence and leave the model with nothing to answer from. This stage
answers the second question the only way it can be answered: build the filtered
context, hand it to the same generation prompt the baseline used, and classify the
reply.

Design choices, all of them there to keep the comparison honest:

* **The same prompt as the baseline.** ``gen_v1_plain``, the measurement prompt, so
  the only thing that differs between two rows of the result table is the context.
  The guarded prompt would bundle a second defence into the measurement.
* **C0 is the reference and it is free.** With no defence the context is the plain
  top-k, so the request is byte-identical to the one the baseline audit already paid
  for and the cache answers it.
* **An empty pool is an abstention by design.** If the filter removes everything the
  model is asked with no context at all, which is the stage-C no-context prompt and
  is already cached for the same questions.
* **Only the test split.** Thresholds, weights and coefficients are fitted on the
  calibration split by the same code path the ablation uses.
"""

from __future__ import annotations

import json
from collections import Counter

from ..config import Config, load_ablation_matrix, load_config
from ..datagen.loader import load_bundle
from ..generation.answer import classify
from ..llm import prompts
from ..llm import mock as mock_responders
from ..llm.client import Call, LLMClient
from ..paths import path as repo_path, rel
from ..signals.load import load_joined, row_key
from ..fusion import calibrate as calibration
from ..fusion.fuse import build_model
from ..fusion.policy import empty_pool_keeps_everything, filter_pool
from .aggregate import pool_index
from .stats import mcnemar

#: Configurations carried forward to the end-to-end stage. One baseline, the best
#: single signal with traces, the judgement pair, the two fusions, and C1 because it
#: is the case where a high clean_rate comes from cutting too much.
DEFENCE_CONFIGS: tuple[str, ...] = ("C0", "C1", "C4", "C7", "C8", "C8l")


def run_defence(
    seeds: list[int] | None = None,
    *,
    configs: tuple[str, ...] | None = None,
    config: Config | None = None,
    k: int | None = None,
    concurrency: int | None = None,
    budget_cny: float | None = None,
    mock: bool = False,
    dry_run: bool = False,
    write: bool = True,
    quiet: bool = False,
) -> dict:
    cfg = config or load_config("default.yaml")
    matrix = load_ablation_matrix()
    wanted = list(configs or DEFENCE_CONFIGS)
    unknown = [name for name in wanted if name not in matrix]
    if unknown:
        raise KeyError(f"unknown ablation config(s): {unknown}")
    top_k = int(k or cfg.get("retrieval.top_k"))
    max_tokens = int(cfg.get("llm.max_tokens_answer"))
    seed_list = seeds or [42]

    results: list[dict] = []
    for seed in seed_list:
        bundle = load_bundle(seed)
        texts = bundle.texts()
        queries = {row["query_id"]: row for row in bundle.queries}
        rows = load_joined(seed)
        if not rows:
            raise RuntimeError(f"no signal table for seed {seed}")
        calib = [row for row in rows if row["split"] == "calib"]
        test_pools = pool_index([row for row in rows if row["split"] == "test"])

        plans: list[dict] = []
        calls: list[Call] = []
        for config_id in wanted:
            spec = matrix[config_id]
            model = build_model(spec.signals, spec.fusion, flip_query_sign=spec.flip_query_sign)
            if spec.fusion == "weighted":
                calibration.fit_alphas(model, calib)
            elif spec.fusion == "logistic":
                calibration.fit_logistic(model, calib)
            threshold = (
                calibration.fit_threshold(model, calib) if spec.fusion != "none" else None
            )
            tau = threshold.chosen if (threshold and spec.fusion != "none") else 0.0
            scores = {row_key(row): model.score(row) for row in rows}

            for key, subset in sorted(test_pools.items()):
                if spec.fusion == "none":
                    pool = empty_pool_keeps_everything(subset, top_k)
                else:
                    pool = filter_pool(subset, scores, tau, top_k)
                query = queries[subset[0]["query_id"]]
                if pool.abstained:
                    user = prompts.CONTROL_USER_NO_CONTEXT.format(question=query["question"])
                else:
                    context = prompts.format_context([texts[doc] for doc in pool.kept])
                    user = prompts.GEN_USER.format(question=query["question"], context=context)
                call = Call(
                    system=prompts.GEN_PLAIN_SYSTEM,
                    user=user,
                    stage=f"defence_{config_id}",
                    max_tokens=max_tokens,
                    extra={"corpus_id": subset[0]["corpus_id"], "query_id": subset[0]["query_id"]},
                )
                calls.append(call)
                plans.append(
                    {
                        "config": config_id,
                        "key": key,
                        "corpus_id": subset[0]["corpus_id"],
                        "query_id": subset[0]["query_id"],
                        "attack_family": subset[0]["attack_family"],
                        "injection": int(subset[0]["injection"]),
                        "fact_id": subset[0]["fact_id"],
                        "query": query,
                        "pool": pool,
                        "call_index": len(calls) - 1,
                    }
                )

        if dry_run:
            client = LLMClient(cfg, concurrency=concurrency, budget_cny=budget_cny, mock=mock, quiet=True)
            unique = {client._cache_key(call) for call in calls}
            cached = sum(1 for key in unique if client._cache_path(key).exists())
            per_config: dict[str, dict] = {}
            for plan in plans:
                call = calls[plan["call_index"]]
                key = client._cache_key(call)
                entry = per_config.setdefault(plan["config"], {"requests": 0, "keys": set()})
                entry["requests"] += 1
                entry["keys"].add(key)
            for config_id, entry in per_config.items():
                entry["unique"] = len(entry["keys"])
                entry["cached"] = sum(1 for key in entry["keys"] if client._cache_path(key).exists())
                entry.pop("keys")
            results.append(
                {
                    "seed": seed,
                    "dry_run": True,
                    "instances": len(plans),
                    "requests": len(calls),
                    "unique": len(unique),
                    "cached": cached,
                    "to_pay": len(unique) - cached,
                    "per_config": per_config,
                }
            )
            if not quiet:
                print(
                    f"  seed {seed}: {len(plans)} generations, {len(unique)} unique prompts, "
                    f"{cached} cached, {len(unique) - cached} to pay"
                )
                for config_id, entry in per_config.items():
                    print(
                        f"    {config_id:<5} {entry['requests']:>5} requests, "
                        f"{entry['unique']:>5} unique, {entry['cached']:>5} cached"
                    )
            continue

        client = LLMClient(
            cfg,
            concurrency=concurrency,
            budget_cny=budget_cny,
            mock=mock,
            mock_responder=mock_responders.last_number,
            quiet=quiet,
        )
        if not quiet:
            print(
                f"  seed {seed}: {len(plans)} generations over {len(wanted)} configurations ...",
                flush=True,
            )
        outcomes = client.complete_many(calls, label="defence    ", progress_every=500)

        labels = Counter()
        records: list[dict] = []
        for plan in plans:
            outcome = outcomes[plan["call_index"]]
            label = (
                classify(outcome.text, plan["query"]["correct_answer"], plan["query"]["target_answer"])
                if outcome.ok
                else "failed"
            )
            pool = plan["pool"]
            records.append(
                {
                    "config": plan["config"],
                    "corpus_id": plan["corpus_id"],
                    "query_id": plan["query_id"],
                    "attack_family": plan["attack_family"],
                    "injection": plan["injection"],
                    "fact_id": plan["fact_id"],
                    "label": label,
                    "answer": outcome.text.strip()[:60],
                    "atr_before": pool.atr_before,
                    "atr_after": pool.atr_after,
                    "filtered_ratio": pool.filtered_ratio,
                    "refilled": int(bool(pool.refilled)),
                    "abstained": int(pool.abstained),
                }
            )
            labels[(plan["config"], label)] += 1

        summary = _summarise(records, wanted)
        results.append(
            {
                "seed": seed,
                "dry_run": False,
                "n_instances": len(
                    {
                        (row["corpus_id"], row["query_id"])
                        for row in records
                        if row["config"] == wanted[0]
                    }
                ),
                "configs": summary,
                "llm": client.summary(),
            }
        )

        if not quiet:
            print(_console(summary, wanted))
        if write:
            _write(seed, records, summary, wanted, top_k)
            results[-1]["report"] = rel(repo_path("reports", f"defence_seed{seed}.md"))
        if not quiet:
            print()

    return {"seeds": results}


def _instance_key(row: dict) -> tuple:
    """One generated answer belongs to one pool, not to one candidate."""
    return (row["corpus_id"], row["query_id"])


def _summarise(records: list[dict], configs: list[str]) -> dict:
    out: dict[str, dict] = {}
    baseline = {}
    for config_id in configs:
        subset = [row for row in records if row["config"] == config_id]
        if not subset:
            continue
        total = len(subset)
        counts = Counter(row["label"] for row in subset)
        entry = {
            "n": total,
            "asr": counts["target"] / total,
            "correct": counts["correct"] / total,
            "unknown": counts["unknown"] / total,
            "other": counts["other"] / total,
            "failed": counts["failed"] / total,
            "atr_after": sum(row["atr_after"] for row in subset) / total,
            "atr_before": sum(row["atr_before"] for row in subset) / total,
            "filtered_ratio": sum(row["filtered_ratio"] for row in subset) / total,
            "refilled": sum(row["refilled"] for row in subset),
            "abstained": sum(row["abstained"] for row in subset),
            "by_family": {},
            "by_injection": {},
        }
        for family in sorted({row["attack_family"] for row in subset}):
            family_rows = [row for row in subset if row["attack_family"] == family]
            family_counts = Counter(row["label"] for row in family_rows)
            entry["by_family"][family] = {
                "n": len(family_rows),
                "asr": family_counts["target"] / len(family_rows),
                "correct": family_counts["correct"] / len(family_rows),
                "unknown": family_counts["unknown"] / len(family_rows),
            }
        for injection in sorted({row["injection"] for row in subset}):
            injection_rows = [row for row in subset if row["injection"] == injection]
            injection_counts = Counter(row["label"] for row in injection_rows)
            entry["by_injection"][str(injection)] = {
                "n": len(injection_rows),
                "asr": injection_counts["target"] / len(injection_rows),
                "correct": injection_counts["correct"] / len(injection_rows),
            }
        out[config_id] = entry
        if config_id == "C0":
            baseline = {_instance_key(row): row["label"] for row in subset}
        # paired comparison against the undefended baseline
        if config_id != "C0" and baseline:
            candidate = {
                _instance_key(row): row["label"]
                for row in subset
                if _instance_key(row) in baseline
            }
            successes = {
                key: int(label == "target") for key, label in candidate.items()
            }
            reference = {key: int(baseline[key] == "target") for key in successes}
            # McNemar on "the attack still works": more target answers under the
            # defence is worse, so only_reference counts as a win for the defence.
            test = mcnemar(reference, successes)
            entry["mcnemar_vs_c0"] = test
            entry["delta_asr"] = out["C0"]["asr"] - entry["asr"]
            entry["delta_correct"] = entry["correct"] - out["C0"]["correct"]
    return out


def _write(seed: int, records: list[dict], summary: dict, configs: list[str], top_k: int) -> None:
    out = repo_path("results")
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"defence_seed{seed}.json"
    json_path.write_text(
        json.dumps({"seed": seed, "configs": summary}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    cells = out / f"defence_seed{seed}_cells.jsonl"
    with cells.open("w", encoding="utf-8", newline="\n") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    md = repo_path("reports", f"defence_seed{seed}.md")
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(_render(seed, summary, configs, top_k), encoding="utf-8", newline="\n")


def _render(seed: int, summary: dict, configs: list[str], top_k: int) -> str:
    lines = [f"# End-to-end defence - seed {seed}", ""]
    lines.append(
        "Every configuration filters the pool, then generates an answer from what is left with the "
        "same prompt the baseline used. `ASR` is the share of instances where the reply matches the "
        "attacker's value; `correct` is the share that matches the genuine answer; `UNKNOWN` is an "
        "abstention."
    )
    lines.append("")
    lines.append("| config | ASR | correct | UNKNOWN | other | ΔASR vs C0 | Δcorrect vs C0 | poison in context | filtered | refilled | abstained |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for config_id in configs:
        entry = summary.get(config_id)
        if not entry:
            continue
        if config_id == "C0":
            delta_asr = delta_correct = "-"
        else:
            delta_asr = f"{entry.get('delta_asr', 0.0):+.3f}"
            delta_correct = f"{entry.get('delta_correct', 0.0):+.3f}"
        lines.append(
            f"| {config_id} | **{entry['asr']:.3f}** | {entry['correct']:.3f} | "
            f"{entry['unknown']:.3f} | {entry['other']:.3f} | {delta_asr} | {delta_correct} | "
            f"{entry['atr_after']:.3f} | {entry['filtered_ratio']:.2f} | "
            f"{entry['refilled']} | {entry['abstained']} |"
        )
    lines.append("")
    lines.append("## Attack success by attack family")
    lines.append("")
    families = sorted({family for entry in summary.values() for family in entry["by_family"]})
    short = [family.split("_")[0] for family in families]
    lines.append("| config | " + " | ".join(short) + " |")
    lines.append("|---|" + "---|" * len(families))
    for config_id in configs:
        entry = summary.get(config_id)
        if not entry:
            continue
        cells = [
            f"{entry['by_family'].get(family, {}).get('asr', 0.0):.2f}" for family in families
        ]
        lines.append(f"| {config_id} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Attack success by injection size")
    lines.append("")
    lines.append("| config | inj=1 | inj=3 |")
    lines.append("|---|---|---|")
    for config_id in configs:
        entry = summary.get(config_id)
        if not entry:
            continue
        cells = [
            f"{entry['by_injection'].get(value, {}).get('asr', 0.0):.2f}" for value in ("1", "3")
        ]
        lines.append(f"| {config_id} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Paired test against the undefended baseline")
    lines.append("")
    lines.append("| config | attack stopped (defence won) | attack started (defence lost) | p |")
    lines.append("|---|---|---|---|")
    for config_id in configs:
        entry = summary.get(config_id)
        if not entry or "mcnemar_vs_c0" not in entry:
            continue
        test = entry["mcnemar_vs_c0"]
        lines.append(
            f"| {config_id} | {test['only_reference']} | {test['only_candidate']} | "
            f"{test['p_value']:.2g} |"
        )
    lines.append("")
    lines.append(
        "A pool counts as 'the attack still works' when the reply matches the attacker's value. "
        "'attack stopped' pools are those where C0 was fooled and this configuration was not; "
        "'attack started' pools are the reverse, i.e. cases where adding a defence made the "
        "attacker succeed. Both can be non-zero, which is the point of reporting them separately."
    )
    lines.append("")
    return "\n".join(lines)


def _console(summary: dict, configs: list[str]) -> str:
    lines = ["", f"{'config':<8}{'ASR':>8}{'correct':>9}{'UNKNOWN':>9}{'ΔASR':>8}{'Δcorrect':>10}"]
    for config_id in configs:
        entry = summary.get(config_id)
        if not entry:
            continue
        delta_asr = "n/a" if config_id == "C0" else f"{entry.get('delta_asr', 0.0):+.3f}"
        delta_correct = "n/a" if config_id == "C0" else f"{entry.get('delta_correct', 0.0):+.3f}"
        lines.append(
            f"{config_id:<8}{entry['asr']:>8.3f}{entry['correct']:>9.3f}{entry['unknown']:>9.3f}"
            f"{delta_asr:>8}{delta_correct:>10}"
        )
    return "\n".join(lines)
