"""Figures for the dashboard. Pure matplotlib on the Agg backend.

No streamlit import, so every figure can be rendered to a PNG and inspected without
a browser. That is the only visual verification available here, and it is also how
`run.py ui-check` proves that the chart code runs over the real data rather than over
an empty frame.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: no display is ever available in this environment

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK = "#1f2933"
MUTED = "#7b8794"
ACCENT = "#b23c17"
COOL = "#2f6f9f"
GRID = "#e4e7eb"


def _style(ax) -> None:
    ax.set_facecolor("white")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def empty(message: str = "No data") -> "plt.Figure":
    figure, ax = plt.subplots(figsize=(6, 2.2), dpi=110)
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", color=MUTED, fontsize=11)
    return figure


def heatmap(
    matrix: list[list[float]],
    row_labels: list[str],
    column_labels: list[str],
    *,
    title: str = "",
    cmap: str = "YlOrRd",
    vmin: float | None = None,
    vmax: float | None = None,
    fmt: str = "{:.2f}",
    annotate: bool = True,
) -> "plt.Figure":
    if not matrix or not row_labels or not column_labels:
        return empty(title or "No data")
    data = np.array(matrix, dtype=float)
    height = max(2.2, 0.42 * len(row_labels) + 1.2)
    width = max(4.6, 0.95 * len(column_labels) + 1.8)
    figure, ax = plt.subplots(figsize=(width, height), dpi=110)
    image = ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(column_labels)))
    ax.set_xticklabels(column_labels, rotation=0, color=INK, fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, color=INK, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    span = (vmax - vmin) if (vmin is not None and vmax is not None) else (data.max() - data.min() or 1.0)
    midpoint = (vmin + span / 2) if vmin is not None else data.mean()
    if annotate:
        for row in range(data.shape[0]):
            for column in range(data.shape[1]):
                value = data[row, column]
                colour = "white" if value > midpoint else INK
                ax.text(
                    column, row, fmt.format(value), ha="center", va="center",
                    color=colour, fontsize=8,
                )
    figure.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    figure.tight_layout()
    return figure


def bars(
    labels: list[str],
    values: list[float],
    *,
    title: str = "",
    ylabel: str = "",
    highlight: set[str] | None = None,
    horizontal: bool = False,
    reference: float | None = None,
    ylim: tuple[float, float] | None = None,
) -> "plt.Figure":
    if not labels or not values:
        return empty(title or "No data")
    highlight = highlight or set()
    height = max(2.2, 0.34 * len(labels) + 1.1)
    figure, ax = plt.subplots(figsize=(7.2, height), dpi=110)
    _style(ax)
    colours = [ACCENT if label in highlight else COOL for label in labels]
    positions = range(len(labels))
    if horizontal:
        ax.barh(list(positions), values, color=colours, height=0.68)
        ax.set_yticks(list(positions))
        ax.set_yticklabels(labels, color=INK, fontsize=9)
        ax.invert_yaxis()
    else:
        ax.bar(list(positions), values, color=colours, width=0.68)
        ax.set_xticks(list(positions))
        ax.set_xticklabels(labels, color=INK, fontsize=9)
    if reference is not None:
        if horizontal:
            ax.axvline(reference, color=MUTED, linewidth=1.0, linestyle="--")
        else:
            ax.axhline(reference, color=MUTED, linewidth=1.0, linestyle="--")
    if ylim:
        # The value axis is horizontal when the bars are, so a 0..1 range must go to
        # x there; putting it on y clips the categories and silently hides most bars.
        if horizontal:
            ax.set_xlim(*ylim)
        else:
            ax.set_ylim(*ylim)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    if ylabel and not horizontal:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    figure.tight_layout()
    return figure


def line(
    series: dict[str, tuple[list[float], list[float]]],
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    reference: tuple[float, float] | None = None,
) -> "plt.Figure":
    if not series:
        return empty(title or "No data")
    figure, ax = plt.subplots(figsize=(7.2, 3.2), dpi=110)
    _style(ax)
    for name, (xs, ys) in series.items():
        ax.plot(xs, ys, label=name, color=ACCENT if "fusion" in name.lower() else COOL, linewidth=1.8)
    if reference:
        ax.axvline(reference[0], color=MUTED, linewidth=1.0, linestyle="--")
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=9)
    figure.tight_layout()
    return figure


def scatter(
    points: list[tuple[float, float, str]],
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
) -> "plt.Figure":
    if not points:
        return empty(title or "No data")
    figure, ax = plt.subplots(figsize=(6.4, 3.6), dpi=110)
    _style(ax)
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    ax.scatter(xs, ys, color=COOL, s=42, alpha=0.85)
    for x, y, label in points:
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8, color=INK)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    figure.tight_layout()
    return figure


def confusion(tp: int, fp: int, fn: int, tn: int, *, title: str = "Confusion matrix") -> "plt.Figure":
    matrix = [[tp, fn], [fp, tn]]
    figure, ax = plt.subplots(figsize=(3.4, 2.8), dpi=110)
    image = ax.imshow(matrix, cmap="Blues")
    labels = [["TP", "FN"], ["FP", "TN"]]
    for row in range(2):
        for column in range(2):
            value = matrix[row][column]
            ax.text(
                column, row, f"{labels[row][column]}\n{value}", ha="center", va="center",
                color="white" if value > max(tp, fp, fn, tn) / 2 else INK, fontsize=10,
            )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["flagged", "kept"], color=INK, fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["poison", "benign"], color=INK, fontsize=9)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    figure.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    figure.tight_layout()
    return figure
