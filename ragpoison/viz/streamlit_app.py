"""The five-tab dashboard. Layout only.

Every number comes from ``data_layer`` and every figure from ``charts``; nothing here
computes a metric. That separation exists because the browser is not available during
development, so the data and chart layers are exercised by the self-test and by
``run.py ui-check``, and only the arrangement is left here.

Two rules this file tries to keep:

* **a label must not promise what the picture does not show** - the per-signal chart
  used to advertise a whole-pool versus hard-subset comparison and plot one series;
* **a number must say what population it covers** - the ledger cards used to show a
  tail window as if it were the project total.
"""

from __future__ import annotations

import math

import streamlit as st

from ragpoison.signals.base import ABLATION_COLUMNS
from ragpoison.viz import charts, data_layer

st.set_page_config(page_title="RAG poisoning detection", layout="wide")

st.title("RAG poisoning detection - evaluation dashboard")
st.caption(
    "Offline. Every figure reads files under results/ and reports/; nothing on this "
    "page spends money unless you explicitly ask the inspector to re-run a query."
)


def _seeds() -> list[int]:
    return data_layer.available_seeds() or [42]


def _sigmoid(value: float) -> float:
    if value < -60.0:
        return 0.0
    if value > 60.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-value))


@st.cache_data(show_spinner=False)
def _totals(seed: int) -> dict:
    return data_layer.ledger_totals()


@st.cache_data(show_spinner=False)
def _sweep(seed: int, config_id: str) -> dict:
    return data_layer.sweep(seed, config_id)


@st.cache_data(show_spinner=False)
def _hard_family(seed: int) -> dict:
    return data_layer.hard_by_family(seed)


with st.sidebar:
    st.header("Scope")
    seed = st.selectbox("Knowledge base (seed)", _seeds())
    run_mode = data_layer.run_report(seed).get("_mode", "-")
    st.caption(
        f"Per-signal and per-family figures come from the **{run_mode}** run of seed {seed}. "
        "Seed 42 and seed 43 are independent corpora with no shared facts; a finding that "
        "holds on both is a finding."
    )

tabs = st.tabs(
    [
        "1 Prompt inspector",
        "2 Detection",
        "3 Signal x attack",
        "4 Per-query inspector",
        "5 Security scan",
    ]
)


# --------------------------------------------------------------------------- #
# Tab 1 - prompt inspector
# --------------------------------------------------------------------------- #
with tabs[0]:
    st.subheader("Prompt inspector")
    totals = _totals(seed)
    if not totals:
        st.info("No ledger yet. Run any paid stage first.")
    else:
        columns = st.columns(4)
        columns[0].metric("calls (whole ledger)", f"{totals['rows']:,}")
        columns[1].metric("cache hit rate (whole ledger)", f"{totals['cache_rate']:.1%}")
        columns[2].metric(
            "paid tokens in / out",
            f"{totals['paid_tokens_in']:,} / {totals['paid_tokens_out']:,}",
        )
        columns[3].metric("errors", totals["errors"])
        st.caption(
            f"Every figure on this row covers all {totals['rows']:,} ledger rows, of which "
            f"{totals['paid']:,} were paid. The token figures count only the paid calls, so "
            f"they are the ones that cost money; counting cached calls too would report "
            f"{totals['tokens_in']:,} input tokens for a fraction of the cost."
        )

        left, right = st.columns([1, 1])
        with left:
            st.markdown("##### Calls by stage (whole ledger)")
            st.dataframe(
                [
                    {"stage": name, "calls": count, "paid": totals["paid_by_stage"].get(name, 0)}
                    for name, count in totals["by_stage"].items()
                ],
                width="stretch",
                hide_index=True,
            )
        with right:
            st.markdown("##### Recent calls")
            recent = data_layer.ledger(400)
            stages = ["all"] + list(
                dict.fromkeys(row.get("stage") for row in recent if row.get("stage"))
            )
            chosen_stage = st.selectbox("filter by stage", stages, index=0)
            only_paid = st.radio(
                "cache", ["all", "paid only", "cache hits only"], horizontal=True, index=0
            )
            filtered = [
                row
                for row in recent
                if (chosen_stage == "all" or row.get("stage") == chosen_stage)
                and (
                    only_paid == "all"
                    or (only_paid == "paid only" and not row.get("cache_hit"))
                    or (only_paid == "cache hits only" and row.get("cache_hit"))
                )
            ]
            st.caption(f"{len(filtered)} of the last {len(recent)} calls match.")
            st.dataframe(
                [
                    {
                        "stage": row.get("stage"),
                        "cached": "hit" if row.get("cache_hit") else "paid",
                        "in": row.get("prompt_tokens"),
                        "out": row.get("completion_tokens"),
                        "ms": row.get("latency_ms"),
                        "tag": str(row.get("tag") or "")[:40],
                    }
                    for row in filtered[-100:]
                ],
                width="stretch",
                hide_index=True,
            )

        st.markdown("#### Rebuild one request and check it against the paid cache")
        st.caption(
            "The ledger stores tokens and cost, not prompt text. Instead of showing a stored "
            "copy, this reconstructs the request from the pipeline and looks the hash up in the "
            "cache. A hit proves the prompt the code produces today is the one that was paid for."
        )
        try:
            from ragpoison.datagen.corpora import iter_query_corpora
            from ragpoison.datagen.loader import load_bundle

            corpora = [corpus["corpus_id"] for corpus in iter_query_corpora(load_bundle(seed))]
        except Exception:  # noqa: BLE001
            corpora = []
        if not corpora:
            st.info("No generated corpus for this seed.")
        else:
            left, middle, right = st.columns([3, 2, 2])
            corpus_id = left.selectbox("pool", corpora, index=0)
            stage = middle.selectbox(
                "stage",
                ["audit_baseline_plain", "s3_answer", "s4_instruction", "s5_relevance"],
            )
            detail = data_layer.query_detail(seed, corpus_id)
            doc_id = None
            if stage != "audit_baseline_plain" and detail.get("rows"):
                doc_id = right.selectbox("document", [row["doc_id"] for row in detail["rows"]])
            if st.button("Rebuild this request", key="rebuild"):
                payload = data_layer.rebuild_prompt(
                    seed, stage, corpus_id=corpus_id, doc_id=doc_id
                )
                if payload.get("error"):
                    st.warning(payload["error"])
                else:
                    state = "HIT" if payload["cache_hit"] else "MISS"
                    st.write(f"cache key `{payload['cache_key'][:16]}...` -> **{state}**")
                    st.code(payload["system"], language="text")
                    st.code(payload["user"], language="text")
                    if payload["cache_hit"]:
                        st.write("stored response")
                        st.code(payload["response"], language="text")
                        st.caption(
                            f"{payload['prompt_tokens']} prompt tokens, "
                            f"{payload['completion_tokens']} completion tokens"
                        )


# --------------------------------------------------------------------------- #
# Tab 2 - detection
# --------------------------------------------------------------------------- #
with tabs[1]:
    st.subheader("Detection")
    rows = [row for row in data_layer.ablation() if row.get("seed") == seed]
    if not rows:
        st.info("No ablation summary. Run `python run.py refusion`.")
    else:
        names = [row["config"] for row in rows]
        chosen = st.selectbox(
            "Configuration", names, index=names.index("C8l") if "C8l" in names else 0
        )
        row = next(item for item in rows if item["config"] == chosen)
        cards = st.columns(6)
        for column, key, label in zip(
            cards,
            ("precision", "recall", "f1", "fpr", "fnr", "auc"),
            ("precision", "recall", "F1", "FPR", "FNR", "AUC"),
        ):
            value = row.get(key)
            column.metric(label, f"{value:.3f}" if isinstance(value, float) else "-")
        st.caption(
            f"clean_rate {row.get('clean_rate', 0):.3f} "
            f"[{row.get('clean_rate_lo', 0):.3f}, {row.get('clean_rate_hi', 0):.3f}] "
            f"over {int(row.get('n_pools_scored') or 0)} pools; threshold "
            f"{row.get('tau', 0):.4f}, false-positive budget 5%."
        )

        left, right = st.columns([1, 2])
        with left:
            st.pyplot(
                charts.confusion(
                    int(row.get("tp") or 0),
                    int(row.get("fp") or 0),
                    int(row.get("fn") or 0),
                    int(row.get("tn") or 0),
                    title=f"{chosen} at its operating threshold",
                ),
                clear_figure=True,
            )
        with right:
            st.pyplot(
                charts.bars(
                    names,
                    [float(item.get("clean_rate") or 0.0) for item in rows],
                    title="Pools with the poison entirely removed (clean_rate)",
                    ylabel="clean_rate",
                    highlight={chosen},
                    horizontal=True,
                    ylim=(0, 1),
                ),
                clear_figure=True,
            )

        st.markdown("#### Per-signal AUC: whole pool against the hard subset")
        report = data_layer.run_report(seed)
        if report:
            signals = list(report.get("auc", {}))
            whole = [float(report["auc"].get(name) or 0.5) for name in signals]
            hard = [float(report["auc_hard"].get(name) or 0.5) for name in signals]
            st.pyplot(
                charts.grouped_bars(
                    [data_layer.SHORT.get(name, name) for name in signals],
                    {"whole pool": whole, "hard subset": hard},
                    title="AUC per signal: whole pool (grey) against poison-versus-evidence (orange)",
                    xlim=(0, 1),
                ),
                clear_figure=True,
            )
            st.caption(
                "The hard subset keeps only the poisoned passage and the clean passages that "
                "actually answered the question, so no trivially irrelevant negative is left. A "
                "signal that merely detects 'this passage answers the question' scores well on "
                "the whole pool and collapses here."
            )
            st.dataframe(
                [
                    {
                        "signal": data_layer.SHORT.get(name, name),
                        "whole pool": round(whole[index], 3),
                        "hard subset": round(hard[index], 3),
                        "difference": round(hard[index] - whole[index], 3),
                    }
                    for index, name in enumerate(signals)
                ],
                width="stretch",
                hide_index=True,
            )
            st.caption(
                f"{report.get('n_hard')} candidates in the hard subset out of "
                f"{report.get('n_rows')} rows. A negative difference means the signal was "
                "being carried by the easy negatives."
            )

        curve = _sweep(seed, chosen)
        if curve:
            left, right = st.columns(2)
            with left:
                st.pyplot(
                    charts.line(
                        {chosen: (curve["taus"], curve["f1"])},
                        title=f"F1 against the threshold - {chosen}",
                        xlabel="threshold",
                        ylabel="F1",
                        reference=(curve["operating"], 0.0),
                    ),
                    clear_figure=True,
                )
            with right:
                st.pyplot(
                    charts.line(
                        {chosen: (curve["recall"], curve["precision"])},
                        title=f"Precision-recall curve - {chosen}",
                        xlabel="recall",
                        ylabel="precision",
                    ),
                    clear_figure=True,
                )
            st.caption(
                f"The dashed line marks the operating threshold (benign false-positive rate held "
                f"at 5%). F1 there is {curve['f1_at_operating']:.3f}; the curve shows what a "
                "different operating point would buy. Both are drawn over every candidate rather "
                "than a sample, so the precision-recall line is continuous."
            )

        st.markdown("#### Cost against detection")
        st.pyplot(
            charts.scatter(
                [
                    (
                        float(
                            data_layer.cost_units(
                                item["signals"].split("+") if item["signals"] != "none" else []
                            )
                        ),
                        float(item.get("clean_rate") or 0.0),
                        item["config"],
                    )
                    for item in rows
                ],
                title="API calls per instance against clean_rate",
                xlabel="calls per instance",
                ylabel="clean_rate",
            ),
            clear_figure=True,
        )
        st.dataframe(
            [
                {
                    "config": item["config"],
                    "signals": item["signals"],
                    "calls per instance": data_layer.cost_units(
                        item["signals"].split("+") if item["signals"] != "none" else []
                    ),
                    "clean_rate": round(float(item.get("clean_rate") or 0.0), 3),
                }
                for item in rows
            ],
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Calls per instance counts S3 (15 per instance) and S5 (9 per batch view) as the "
            "paid signals and S4 as one amortised call; S1, S2 and S6 are free. Read this "
            "against the end-to-end result: the cheapest configuration that keeps the evidence "
            "is not the one with the highest clean_rate."
        )


# --------------------------------------------------------------------------- #
# Tab 3 - signal x attack
# --------------------------------------------------------------------------- #
with tabs[2]:
    st.subheader("Which signal responds to which attack")
    payload = data_layer.signal_attack(seed)
    if not payload:
        st.info("No run report. Run `python run.py full` or `replicate`.")
    else:
        families = list(payload["per_family_auc"])
        signals = list(next(iter(payload["per_family_auc"].values())).keys())
        hard = _hard_family(seed)
        short = [data_layer.SHORT.get(name, name) for name in signals]
        family_short = [family.split("_")[0] for family in families]

        st.caption(
            "The whole-pool view is the flattering one: most negatives cannot answer the "
            "question at all, so a signal that only detects 'this passage answers the question' "
            "scores well. The hard-subset view keeps only poison and the clean passages that "
            "answered, which is the comparison that decides whether a signal is useful."
        )
        view = st.radio(
            "View",
            [
                "whole pool (easy negatives)",
                "hard subset (poison against real evidence)",
                "difference: hard minus whole",
            ],
            horizontal=True,
            index=1,
        )

        if view.startswith("whole"):
            matrix = [
                [float(payload["per_family_auc"][family].get(name) or 0.5) for family in families]
                for name in signals
            ]
            st.pyplot(
                charts.heatmap(
                    matrix, short, family_short,
                    title="AUC per signal, per attack family - whole pool",
                    vmin=0.3, vmax=1.0, cmap="RdYlGn",
                ),
                clear_figure=True,
            )
        elif view.startswith("hard"):
            if not hard:
                st.info("No hard-subset data for this seed.")
            else:
                matrix = [
                    [float(hard[family].get(name) or 0.5) for family in families] for name in signals
                ]
                st.pyplot(
                    charts.heatmap(
                        matrix, short, family_short,
                        title="AUC per signal, per attack family - hard subset",
                        vmin=0.3, vmax=1.0, cmap="RdYlGn",
                    ),
                    clear_figure=True,
                )
                st.caption(
                    "Hard subset = the poisoned passage against the clean passages that answered "
                    "the question. S1, S3 and S3b fall to chance or below, while S4 stays at 1.00 "
                    "on exactly the families that write traces."
                )
        else:
            if not hard:
                st.info("No hard-subset data for this seed.")
            else:
                matrix = [
                    [
                        float(hard[family].get(name) or 0.5)
                        - float(payload["per_family_auc"][family].get(name) or 0.5)
                        for family in families
                    ]
                    for name in signals
                ]
                st.pyplot(
                    charts.heatmap(
                        matrix, short, family_short,
                        title="Change from whole pool to hard subset (negative = collapse)",
                        vmin=-0.7, vmax=0.7, cmap="RdYlGn", fmt="{:+.2f}",
                        # A diverging scale is dark at both ends, so the midpoint rule
                        # leaves white text on the palest cells. This is a new chart, so
                        # it gets the readable rule; the two older heatmaps keep theirs.
                        label_contrast="auto",
                    ),
                    clear_figure=True,
                )
                st.caption("Red means the signal looked better than it is once the easy negatives go.")

        st.markdown("#### Mean signal value by label")
        labels = list(payload["per_label"])
        st.dataframe(
            [
                {
                    "signal": data_layer.SHORT.get(name, name),
                    **{
                        label: round(float(payload["per_label"][label].get(name) or 0.0), 3)
                        for label in labels
                    },
                }
                for name in signals
            ],
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "The distractor column is the useful control: it should be the least suspicious "
            "everywhere, otherwise the signal is measuring irrelevance rather than poison."
        )


# --------------------------------------------------------------------------- #
# Tab 4 - per-query inspector
# --------------------------------------------------------------------------- #
with tabs[3]:
    st.subheader("Per-query inspector")
    labels = data_layer.query_index(seed, split="test") or data_layer.query_index(seed)
    if not labels:
        st.info("No signal table for this seed.")
    else:
        label = st.selectbox("Pool (test split)", labels, index=0)
        detail = data_layer.query_detail(seed, label)
        if not detail:
            st.warning("That pool could not be loaded.")
        else:
            st.write(f"**{detail['question']}**")
            st.caption(
                f"genuine answer `{detail['correct_answer']}` / attacker target "
                f"`{detail['target_answer']}` / split `{detail['split']}` / family "
                f"`{detail['attack_family']}` / injection {detail['injection']}"
            )

            fitted = data_layer.ablation_details().get((seed, "C8l"), {})
            coefficients = fitted.get("coefficients") or {}
            intercept = float(fitted.get("intercept") or 0.0)
            fitted_tau = float(fitted.get("tau_quantile") or 0.5)

            st.markdown("#### Two scoring rules, side by side")
            if coefficients:
                st.caption(
                    f"**Manual** uses the sliders and does not correspond to any reported row. "
                    f"**C8l fitted** is the logistic model of the C8l ablation row, with its own "
                    f"intercept {intercept:+.3f} and its own threshold {fitted_tau:.4f}: that "
                    "column is the one a reported number corresponds to."
                )
            else:
                st.caption(
                    "**Manual** uses the sliders and does not correspond to any reported row. "
                    "The fitted C8l model is unavailable for this seed."
                )

            controls = st.columns(7)
            weights = {}
            for column, name, default in zip(
                controls,
                ("s1", "s2", "s3", "s4", "s5", "s6", "s7"),
                (0.5, 1.0, 1.5, 1.0, 1.5, 0.5, 0.0),
            ):
                weights[name] = column.slider(name.upper(), 0.0, 2.0, default, 0.5)
            threshold = st.slider("manual threshold", 0.0, 1.0, 0.5, 0.01)

            scored = data_layer.fuse_rows(detail["rows"], weights, threshold)
            for item in scored:
                if coefficients:
                    linear = intercept + sum(
                        float(weight) * float(item.get(ABLATION_COLUMNS[name][1], 0.0))
                        for name, weight in coefficients.items()
                    )
                    item["fitted"] = _sigmoid(linear)
                    item["fitted_flagged"] = item["fitted"] > fitted_tau
                else:
                    item["fitted"] = None
                    item["fitted_flagged"] = False

            st.dataframe(
                [
                    {
                        "rank": item["rank"],
                        "doc": item["doc_id"],
                        "label": item["label"],
                        "S1": round(item["s1"], 3),
                        "S2": round(item["s2"], 3),
                        "S3": round(item["s3"], 3),
                        "S4": round(item["s4"], 3),
                        "S5": round(item["s5"], 3),
                        "S6": round(item["s6"], 3),
                        "manual": round(item["score"], 3),
                        "manual flag": "yes" if item["flagged"] else "",
                        "C8l score": round(item["fitted"], 3) if item["fitted"] is not None else "-",
                        "C8l flag": "yes" if item["fitted_flagged"] else "",
                        "extracted": item["answer"],
                    }
                    for item in scored
                ],
                width="stretch",
                hide_index=True,
            )

            kept = [item for item in scored if not item["flagged"]][:5]
            kept_fitted = [item for item in scored if not item["fitted_flagged"]][:5]
            st.markdown("#### What the generator would see")
            st.caption(
                f"manual rule: {sum(1 for item in scored if item['flagged'])} of {len(scored)} "
                f"flagged, {sum(1 for item in kept if item['is_attack'])} poisoned passage(s) "
                f"surviving | C8l rule: {sum(1 for item in scored if item['fitted_flagged'])} "
                f"flagged, {sum(1 for item in kept_fitted if item['is_attack'])} poisoned "
                "passage(s) surviving"
            )
            for item in kept:
                with st.expander(f"[{item['rank']}] {item['doc_id']} ({item['label']})"):
                    st.write(item["text"])

            st.markdown("#### What the defence actually answered for this pool")
            answers = data_layer.defence_pool(seed, detail["corpus_id"], detail["query_id"])
            if not answers:
                st.info(
                    "No end-to-end generation for this pool. The defence stage covers the test "
                    "split only."
                )
            else:
                st.dataframe(
                    [
                        {
                            "config": item.get("config"),
                            "answer": str(item.get("answer") or "")[:60],
                            "outcome": item.get("label"),
                            "poison in context": round(float(item.get("atr_after") or 0.0), 3),
                        }
                        for item in answers
                    ],
                    width="stretch",
                    hide_index=True,
                )
                st.caption(
                    "`outcome` classifies the generated answer: `target` means the attacker won, "
                    "`correct` means the genuine answer survived, `unknown` is a refusal. This is "
                    "the end-to-end table the specification asks for at the bottom of this page."
                )

            st.markdown("#### Live re-run (spends money)")
            st.caption(
                "The signals above are already paid for. Re-running this one pool would ask the "
                "model again, about 39 calls for one instance. Nothing leaves the machine unless "
                "you tick the box and press the button."
            )
            accepted = st.checkbox("I accept the spend for this single instance", key="accept")
            budget = st.number_input("Hard call budget", min_value=1, max_value=60, value=60, step=1)
            if st.button("Re-run this instance", disabled=not accepted, key="rerun"):
                st.warning(
                    f"Budget guard: at most {int(budget)} calls. The command equivalent is "
                    f"`python run.py signals --stage llm --seed {seed} --query "
                    f"{detail['query_id']}`."
                )
                st.info(
                    "Deliberately not wired to the API from inside the dashboard. The pipeline "
                    "owns the cache, the retry policy and the ledger; a button that bypasses them "
                    "would produce numbers the rest of the project cannot audit."
                )


# --------------------------------------------------------------------------- #
# Tab 5 - security scan
# --------------------------------------------------------------------------- #
with tabs[4]:
    st.subheader("Security scan (periodic corpus audit)")
    strategies = data_layer.scan_strategies()
    history = data_layer.scans()
    size = data_layer.corpus_size(seed)
    st.caption(
        f"Corpus size for this seed: {size} chunks. The strategies below are the configured "
        "policies P1-P5; this page fills in when the scan stage runs."
    )
    if strategies:
        st.dataframe(
            [
                {
                    "strategy": item.get("name"),
                    "mode": item.get("strategy"),
                    "batch size": item.get("batch_size"),
                    "coarse threshold": item.get("coarse_threshold"),
                    "shards": item.get("shard_count"),
                    "budget tokens": item.get("budget_tokens_per_scan"),
                }
                for item in strategies
            ],
            width="stretch",
            hide_index=True,
        )
    if not history:
        st.info(
            "No scan has been run yet. The scan stage (WP9) writes results/scans/index.json, and "
            "this page then shows coverage, suspicious chunks and the cost-coverage curve."
        )
    else:
        covered = sum(int(item.get("covered", 0)) for item in history)
        st.progress(min(1.0, covered / size) if size else 0.0, text=f"coverage {covered}/{size}")
        st.dataframe(history, width="stretch", hide_index=True)
        rounds = -(-size // max(1, int(history[-1].get("covered", 1)))) if history else None
        if rounds:
            st.caption(
                f"At the last round's coverage, about {rounds} rounds would cover the corpus."
            )
