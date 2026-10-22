"""The five-tab dashboard. Layout only.

Every number on these pages comes from ``data_layer`` and every figure from
``charts``; nothing here computes a metric. That separation exists because the
browser is not available during development: the data and chart layers are exercised
by the self-test and by ``run.py ui-check``, so what remains here is only arrangement.
"""

from __future__ import annotations

import streamlit as st

from ragpoison.viz import charts, data_layer

st.set_page_config(page_title="RAG poisoning detection", layout="wide")

st.title("RAG poisoning detection - evaluation dashboard")
st.caption(
    "Offline. Every figure reads files under results/ and reports/; nothing on this "
    "page spends money unless you explicitly ask the inspector to re-run a query."
)


def _seeds() -> list[int]:
    return data_layer.available_seeds() or [42]


with st.sidebar:
    st.header("Scope")
    seed = st.selectbox("Knowledge base (seed)", _seeds())
    mode = st.selectbox("Run", data_layer.available_runs(seed) or ["-"])
    st.caption(
        "Seed 42 and seed 43 are independent corpora generated with no shared facts. "
        "A finding that holds on both is a finding."
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
# Tab 1
# --------------------------------------------------------------------------- #
with tabs[0]:
    st.subheader("Prompt inspector")
    rows = data_layer.ledger(400)
    summary = data_layer.ledger_summary(rows)
    if not rows:
        st.info("No ledger yet. Run any paid stage first.")
    else:
        columns = st.columns(4)
        columns[0].metric("calls (last 400)", summary["rows"])
        columns[1].metric("cache hit rate", f"{summary['cache_rate']:.1%}")
        columns[2].metric("tokens in / out", f"{summary['tokens_in']:,} / {summary['tokens_out']:,}")
        columns[3].metric("errors", summary["errors"])
        st.caption(
            f"{summary['paid']} of these were paid and {summary['cache_hits']} were served "
            "from the answer cache."
        )
        with st.expander("Calls by stage"):
            st.write(summary["by_stage"])
        stage_counts = ", ".join(f"{name}: {count}" for name, count in summary["by_stage"].items())
        st.caption(stage_counts)

        st.markdown("#### Rebuild one request and check it against the paid cache")
        st.caption(
            "The ledger stores tokens and cost, not prompt text. Instead of showing a "
            "stored copy, this reconstructs the request from the pipeline and looks the "
            "hash up in the cache. A hit proves the prompt the code produces today is the "
            "one that was paid for."
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
            detail = data_layer.query_detail(
                seed, corpus_id.replace("_q_", "_inj1_", 1) if False else corpus_id
            )
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
# Tab 2
# --------------------------------------------------------------------------- #
with tabs[1]:
    st.subheader("Detection")
    rows = [row for row in data_layer.ablation() if row.get("seed") == seed]
    if not rows:
        st.info("No ablation summary. Run `python run.py refusion`.")
    else:
        names = [row["config"] for row in rows]
        chosen = st.selectbox("Configuration", names, index=names.index("C8l") if "C8l" in names else 0)
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
            f"over {int(row.get('n_pools_scored') or 0)} pools; "
            f"threshold {row.get('tau', 0):.4f}, FPR budget 5%."
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

        curve = data_layer.sweep(seed, chosen)
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
                    charts.scatter(
                        list(zip(curve["recall"], curve["precision"], [""] * len(curve["recall"]))),
                        title=f"Precision-recall curve - {chosen}",
                        xlabel="recall",
                        ylabel="precision",
                    ),
                    clear_figure=True,
                )
            st.caption(
                f"The dashed line marks the operating threshold (benign false-positive "
                f"rate held at 5%). F1 there is {curve['f1_at_operating']:.3f}; the curve "
                "shows what a different operating point would buy."
            )

        st.markdown("#### Per-signal AUC on this seed")
        report = data_layer.run_report(seed)
        if report:
            signals = list(report.get("auc", {}))
            st.pyplot(
                charts.bars(
                    [data_layer.SHORT.get(name, name) for name in signals],
                    [float(report["auc"][name] or 0.5) for name in signals],
                    title="AUC of each signal alone (whole pool, grey) and on the hard subset (coloured)",
                    ylabel="AUC",
                    horizontal=True,
                    reference=0.5,
                ),
                clear_figure=True,
            )
            st.caption(
                "Hard subset = the poisoned passage against the clean passages that actually "
                "answered the question. A signal that only detects 'this passage answers the "
                "question' scores well on the whole pool and collapses here."
            )
            hard = report.get("auc_hard", {})
            st.write({data_layer.SHORT.get(name, name): round(value, 3) for name, value in hard.items() if value})

        st.markdown("#### Cost against detection")
        details = data_layer.ablation_details()
        st.pyplot(
            charts.scatter(
                [
                    (
                        float(data_layer.cost_units(item["signals"].split("+") if item["signals"] != "none" else [])),
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
        st.caption(
            "Calls per instance counts S3 (15 per instance) and S5 (9 per batch view) as "
            "the paid signals and S4 as one amortised call; the free signals cost nothing. "
            "The trade-off is real: the cheapest configuration on the left of the plot is "
            "also the one that keeps the evidence."
        )


# --------------------------------------------------------------------------- #
# Tab 3
# --------------------------------------------------------------------------- #
with tabs[2]:
    st.subheader("Which signal responds to which attack")
    payload = data_layer.signal_attack(seed)
    if not payload:
        st.info("No run report. Run `python run.py full` or `replicate`.")
    else:
        signals = list(payload["per_family_auc"] and next(iter(payload["per_family_auc"].values())).keys())
        families = list(payload["per_family_auc"])
        view = st.radio(
            "View",
            ["AUC by family", "mean score by label"],
            horizontal=True,
            help="AUC is computed per family over that family's pools; the mean view shows "
            "the raw signal values so a saturated or constant signal is visible.",
        )
        if view == "AUC by family":
            matrix = [
                [float(payload["per_family_auc"][family].get(name) or 0.5) for family in families]
                for name in signals
            ]
            st.pyplot(
                charts.heatmap(
                    matrix,
                    [data_layer.SHORT.get(name, name) for name in signals],
                    [family.split("_")[0] for family in families],
                    title="AUC of each signal, per attack family",
                    vmin=0.3,
                    vmax=1.0,
                    cmap="RdYlGn",
                ),
                clear_figure=True,
            )
            st.caption(
                "Read down a column to see which attack defeats which signal, and across a row "
                "to see how far one signal travels. S4 is at chance on the stealth families and "
                "perfect on the trace-carrying ones; that split is the argument for fusing."
            )
        else:
            labels = list(payload["per_label"])
            matrix = [
                [float(payload["per_label"][label].get(name) or 0.0) for label in labels]
                for name in signals
            ]
            st.pyplot(
                charts.heatmap(
                    matrix,
                    [data_layer.SHORT.get(name, name) for name in signals],
                    labels,
                    title="Mean signal value by label",
                    vmin=0.0,
                    vmax=1.0,
                ),
                clear_figure=True,
            )
            st.caption(
                "The distractor row is the useful control: it should be the least suspicious "
                "everywhere, otherwise the signal is measuring irrelevance rather than poison."
            )


# --------------------------------------------------------------------------- #
# Tab 4
# --------------------------------------------------------------------------- #
with tabs[3]:
    st.subheader("Per-query inspector")
    labels = data_layer.query_index(seed)
    if not labels:
        st.info("No signal table for this seed.")
    else:
        label = st.selectbox("Pool", labels, index=0)
        detail = data_layer.query_detail(seed, label)
        if not detail:
            st.warning("That pool could not be loaded.")
        else:
            st.write(f"**{detail['question']}**")
            st.caption(
                f"genuine answer `{detail['correct_answer']}` / attacker target "
                f"`{detail['target_answer']}` / split `{detail['split']}`"
            )
            st.markdown("#### Weights and threshold")
            controls = st.columns(7)
            weights = {}
            for column, name, default in zip(
                controls, ("s1", "s2", "s3", "s4", "s5", "s6", "s7"),
                (0.5, 1.0, 1.5, 1.0, 1.5, 0.5, 0.0),
            ):
                weights[name] = column.slider(name.upper(), 0.0, 2.0, default, 0.5)
            threshold = st.slider("threshold", 0.0, 1.0, 0.5, 0.01)
            st.caption(
                "These weights only drive the score shown here. They are not the fitted "
                "weights of any ablation row, and moving them changes nothing else."
            )
            scored = data_layer.fuse_rows(detail["rows"], weights, threshold)
            table = [
                {
                    "rank": row["rank"],
                    "doc": row["doc_id"],
                    "label": row["label"],
                    "S1": round(row["s1"], 3),
                    "S2": round(row["s2"], 3),
                    "S3": round(row["s3"], 3),
                    "S4": round(row["s4"], 3),
                    "S5": round(row["s5"], 3),
                    "S6": round(row["s6"], 3),
                    "score": round(row["score"], 3),
                    "flagged": "yes" if row["flagged"] else "",
                    "extracted": row["answer"],
                }
                for row in scored
            ]
            st.dataframe(table, width="stretch", hide_index=True)
            kept = [row for row in scored if not row["flagged"]][:5]
            st.markdown("#### What the generator would see")
            st.caption(
                f"{sum(1 for row in scored if row['flagged'])} of {len(scored)} candidates are "
                f"flagged; the surviving top-5 contains "
                f"{sum(1 for row in kept if row['is_attack'])} poisoned passage(s)."
            )
            for row in kept:
                with st.expander(f"[{row['rank']}] {row['doc_id']} ({row['label']})"):
                    st.write(row["text"])
            if not kept:
                st.warning("The filter would empty the pool, so the model would be asked with no context.")

            st.markdown("#### Live re-run (spends money)")
            st.caption(
                "The stored signals above are already paid for. Re-running this one pool asks "
                "the model again: about 39 calls for one instance. Nothing leaves the machine "
                "unless you tick the box and press the button."
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
                    "Deliberately not wired to the API inside the dashboard. The measurement "
                    "pipeline owns the cache, the retry policy and the ledger; a button that "
                    "bypasses them would produce numbers the rest of the project cannot audit."
                )


# --------------------------------------------------------------------------- #
# Tab 5
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
            "No scan has been run yet. The scan stage (WP9) writes results/scans/index.json, "
            "and this page then shows coverage, suspicious chunks and the cost-coverage curve."
        )
    else:
        covered = sum(int(item.get("covered", 0)) for item in history)
        st.progress(min(1.0, covered / size) if size else 0.0, text=f"coverage {covered}/{size}")
        st.dataframe(history, width="stretch", hide_index=True)
        rounds = -(-size // max(1, int(history[-1].get("covered", 1)))) if history else None
        if rounds:
            st.caption(f"At the last round's coverage, about {rounds} rounds would cover the corpus.")
