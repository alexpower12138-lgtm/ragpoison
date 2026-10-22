"""Every figure the dashboard draws, produced by one command.

This module exists because of a gap that was easy to miss: the report's most
persuasive charts - the per-signal comparison against the hard subset, the collapse
heatmap, the precision-recall curve - were being rendered by one-off scripts during
development and committed nowhere. ``figures/`` is gitignored, so a reader who cloned
the repository could not regenerate them at all.

``python run.py figures`` now writes all of them from the stored tables, so the report's
pictures are as reproducible as its numbers.
"""

from __future__ import annotations

from ..paths import path as repo_path, rel
from . import charts, data_layer


def export_all(seed: int = 42, config_id: str = "C8l") -> list[str]:
    """Render every report figure for one seed and return their paths."""
    target = repo_path("figures")
    target.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def save(figure, name: str) -> None:
        path = target / name
        figure.savefig(path, bbox_inches="tight")
        written.append(rel(path))

    rows = [row for row in data_layer.ablation() if row.get("seed") == seed]
    if rows:
        best = max(rows, key=lambda row: float(row.get("clean_rate") or 0.0))
        save(
            charts.bars(
                [row["config"] for row in rows],
                [float(row.get("clean_rate") or 0.0) for row in rows],
                title=f"Pools with the poison entirely removed, seed {seed}",
                ylabel="clean_rate",
                highlight={best["config"]},
                horizontal=True,
                ylim=(0, 1),
            ),
            f"fig_ablation_clean_rate_seed{seed}.png",
        )

        chosen = next((row for row in rows if row["config"] == config_id), best)
        save(
            charts.confusion(
                int(chosen.get("tp") or 0),
                int(chosen.get("fp") or 0),
                int(chosen.get("fn") or 0),
                int(chosen.get("tn") or 0),
                title=f"{chosen['config']} at its operating threshold",
            ),
            f"fig_confusion_{config_id}_seed{seed}.png",
        )

        save(
            charts.scatter(
                [
                    (
                        float(
                            data_layer.cost_units(
                                row["signals"].split("+") if row["signals"] != "none" else []
                            )
                        ),
                        float(row.get("clean_rate") or 0.0),
                        row["config"],
                    )
                    for row in rows
                ],
                title=f"API calls per instance against clean_rate, seed {seed}",
                xlabel="calls per instance",
                ylabel="clean_rate",
            ),
            f"fig_cost_scatter_seed{seed}.png",
        )

    report = data_layer.run_report(seed)
    if report:
        signals = list(report.get("auc", {}))
        save(
            charts.grouped_bars(
                [data_layer.SHORT.get(name, name) for name in signals],
                {
                    "whole pool": [float(report["auc"].get(name) or 0.5) for name in signals],
                    "hard subset": [float(report["auc_hard"].get(name) or 0.5) for name in signals],
                },
                title="AUC per signal: whole pool against poison-versus-evidence",
                xlim=(0, 1),
            ),
            f"fig_signal_auc_grouped_seed{seed}.png",
        )

        families = list(report.get("per_family_auc", {}))
        if families:
            short = [data_layer.SHORT.get(name, name) for name in signals]
            family_short = [family.split("_")[0] for family in families]
            whole = [
                [float(report["per_family_auc"][family].get(name) or 0.5) for family in families]
                for name in signals
            ]
            save(
                charts.heatmap(
                    whole, short, family_short,
                    title="AUC per signal, per attack family - whole pool",
                    vmin=0.3, vmax=1.0, cmap="RdYlGn",
                ),
                f"fig_signal_attack_whole_seed{seed}.png",
            )
            hard = data_layer.hard_by_family(seed)
            if hard:
                hard_matrix = [
                    [float(hard[family].get(name) or 0.5) for family in families]
                    for name in signals
                ]
                save(
                    charts.heatmap(
                        hard_matrix, short, family_short,
                        title="AUC per signal, per attack family - hard subset",
                        vmin=0.3, vmax=1.0, cmap="RdYlGn",
                    ),
                    f"fig_signal_attack_hard_seed{seed}.png",
                )
                difference = [
                    [
                        hard_matrix[row][column] - whole[row][column]
                        for column in range(len(families))
                    ]
                    for row in range(len(signals))
                ]
                save(
                    charts.heatmap(
                        difference, short, family_short,
                        title="Change from whole pool to hard subset (negative = collapse)",
                        vmin=-0.7, vmax=0.7, cmap="RdYlGn", fmt="{:+.2f}",
                        label_contrast="auto",
                    ),
                    f"fig_signal_attack_difference_seed{seed}.png",
                )

    curve = data_layer.sweep(seed, config_id)
    if curve:
        save(
            charts.line(
                {config_id: (curve["taus"], curve["f1"])},
                title=f"F1 against the threshold - {config_id}, seed {seed}",
                xlabel="threshold",
                ylabel="F1",
                reference=(curve["operating"], 0.0),
            ),
            f"fig_f1_threshold_seed{seed}.png",
        )
        save(
            charts.line(
                {config_id: (curve["recall"], curve["precision"])},
                title=f"Precision-recall curve - {config_id}, seed {seed}",
                xlabel="recall",
                ylabel="precision",
            ),
            f"fig_precision_recall_seed{seed}.png",
        )

    index_path = repo_path("results", "scans", "index.json")
    if index_path.exists():
        import json

        entries = json.loads(index_path.read_text(encoding="utf-8"))
        # Total tokens against corpus recall, not tokens per chunk: a policy that scans
        # a fifth of the corpus at the same per-chunk price is cheap in total, and the
        # per-chunk view hides the coverage dimension that H4 is actually about.
        points = [
            (
                float(entry.get("tokens_used") or 0.0) / 1000.0,
                float(entry.get("recall") or 0.0),
                str(entry.get("policy") or entry.get("scan_id")),
            )
            for entry in entries
            if not entry.get("mock")
        ]
        if points:
            save(
                charts.scatter(
                    points,
                    title="Corpus audit: total cost against corpus recall",
                    xlabel="thousand tokens spent",
                    ylabel="recall of poisoned chunks",
                    leader=True,
                ),
                f"fig_scan_cost_coverage_seed{seed}.png",
            )

    return written
