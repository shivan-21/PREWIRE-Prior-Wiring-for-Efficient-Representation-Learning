"""Figures and tables from the evaluation CSVs.

    python scripts/plot_results.py --seed 0

Reads ``results_and_figs/tables/{finetune,linear_probe}_s<seed>.csv`` and writes, dated:

- ``data_efficiency_finetune_<date>.pdf``: test accuracy vs labelled examples (fine-tuning).
- ``data_efficiency_probe_<date>.pdf``: the same for a linear probe on the last layer.
- ``layer_probe_<date>.pdf``: probe accuracy per layer, at a few labelled-set sizes.
- ``tables/data_equivalence_s<seed>.csv``: labels each prior is worth, vs random init.

Colours are fixed per model (never by rank): LPL blue, SimCLR orange, Oja aqua (the first three
slots of the dataviz reference palette, validated colour-blind safe as a set) and random init as a
grey dashed baseline. Every line is also labelled directly.
"""

import argparse
import csv
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.transforms import ScaledTranslation

from src.analysis.data_equivalence import data_equivalence_table
from src.utils.paths import REPO_ROOT

STYLE = {  # model -> (label, colour, linestyle)
    "lpl": ("LPL", "#2a78d6", "-"),
    "simclr": ("SimCLR", "#eb6834", "-"),
    "oja": ("Oja", "#1baf7a", "-"),
    "random": ("Random init", "#52514e", "--"),
}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LAST_LAYER = 8


def setup_axes(ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_curves(ax: plt.Axes, df: pd.DataFrame, x: str, y: str) -> None:
    """One line per model (in STYLE order), direct-labelled at its right end."""
    ends = []
    for model, (label, colour, style) in STYLE.items():
        rows = df[df.model == model].sort_values(x)
        if rows.empty:
            continue
        ax.plot(
            rows[x],
            rows[y] * 100,
            style,
            color=colour,
            linewidth=1.6,
            marker="o",
            markersize=4,
            label=label,
        )
        ends.append([label, rows[x].iloc[-1], rows[y].iloc[-1] * 100])
    label_ends(ax, ends)


def label_ends(ax: plt.Axes, ends: list[list], min_gap: float = 0.07) -> None:
    """Direct labels at line ends, nudged apart vertically so they never overlap.

    ``min_gap`` is the minimum spacing as a fraction of the y-axis range.
    """
    low, high = ax.get_ylim()
    gap = min_gap * (high - low)
    ends.sort(key=lambda e: e[2])
    placed = [e[2] for e in ends]
    for _ in range(50):  # relax: push neighbours apart symmetrically until spaced
        moved = False
        for i in range(1, len(placed)):
            overlap = gap - (placed[i] - placed[i - 1])
            if overlap > 1e-9:
                placed[i - 1] -= overlap / 2
                placed[i] += overlap / 2
                moved = True
        if not moved:
            break
    shift = ScaledTranslation(6 / 72, 0, ax.figure.dpi_scale_trans)  # 6 pt right of the end
    for (label, x, _), y_text in zip(ends, placed, strict=True):
        ax.text(
            x,
            y_text,
            label,
            transform=ax.transData + shift,
            va="center",
            fontsize=8,
            color=TEXT,
            clip_on=False,
        )


def data_efficiency_figure(df: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    plot_curves(ax, df, "labeled_n", "test_acc")
    ax.set_xscale("log")
    setup_axes(ax, "Labelled training examples (log scale)", "Test accuracy (%)")
    ax.set_title(title, fontsize=10, color=TEXT, loc="left", pad=22)
    ax.legend(
        frameon=False,
        fontsize=8,
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        ncol=4,
        borderaxespad=0.2,
    )
    ax.set_xlim(right=ax.get_xlim()[1] * 2.5)  # room for the direct labels
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"), format="pdf")
    fig.savefig(path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def layer_figure(probe: pd.DataFrame, counts: list[int], path: Path) -> None:
    fig, axes = plt.subplots(1, len(counts), figsize=(3.2 * len(counts), 3.4), sharey=True)
    for ax, n in zip(axes, counts, strict=True):
        plot_curves(ax, probe[probe.labeled_n == n], "layer", "test_acc")
        setup_axes(ax, "Layer", "Test accuracy (%)" if ax is axes[0] else "")
        ax.set_xticks(range(1, LAST_LAYER + 1))
        ax.set_title(f"{n:,} labels", fontsize=9, color=TEXT, loc="left")
        ax.set_xlim(0.5, LAST_LAYER + 1.9)  # room for the direct labels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="upper right", ncol=4)
    fig.suptitle(
        "Linear probe on each layer's features", fontsize=10, color=TEXT, x=0.01, ha="left"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path.with_suffix(".pdf"), format="pdf")
    fig.savefig(path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def equivalence_rows(df: pd.DataFrame, protocol: str) -> list[dict]:
    base = df[df.model == "random"].sort_values("labeled_n")
    rows = []
    for model in ("lpl", "simclr", "oja"):
        m = df[df.model == model].sort_values("labeled_n")
        if m.empty or base.empty:
            continue
        for row in data_equivalence_table(
            m.labeled_n.tolist(),
            m.test_acc.tolist(),
            base.labeled_n.tolist(),
            base.test_acc.tolist(),
        ):
            rows.append({"protocol": protocol, "model": model, **row})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layer_counts", type=int, nargs="+", default=[100, 1000, 50000])
    parser.add_argument("--tables", default=str(REPO_ROOT / "results_and_figs" / "tables"))
    parser.add_argument("--figures", default=str(REPO_ROOT / "results_and_figs" / "figures"))
    args = parser.parse_args()

    tables, figures = Path(args.tables), Path(args.figures)
    stamp = f"{date.today():%Y%m%d}"
    equivalence = []

    finetune_csv = tables / f"finetune_s{args.seed}.csv"
    if finetune_csv.exists():
        ft = pd.read_csv(finetune_csv).drop_duplicates(["model", "labeled_n"], keep="last")
        data_efficiency_figure(
            ft, "Fine-tuning the whole network", figures / f"data_efficiency_finetune_{stamp}"
        )
        equivalence += equivalence_rows(ft, "finetune")

    probe_csv = tables / f"linear_probe_s{args.seed}.csv"
    if probe_csv.exists():
        probe = pd.read_csv(probe_csv).drop_duplicates(["model", "layer", "labeled_n"], keep="last")
        last = probe[probe.layer == LAST_LAYER]
        data_efficiency_figure(
            last,
            "Linear probe on frozen last-layer features",
            figures / f"data_efficiency_probe_{stamp}",
        )
        layer_figure(probe, args.layer_counts, figures / f"layer_probe_{stamp}")
        equivalence += equivalence_rows(last, "probe_last_layer")

    if equivalence:
        out = tables / f"data_equivalence_s{args.seed}.csv"
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(equivalence[0]))
            writer.writeheader()
            writer.writerows(equivalence)
        print(f"wrote {out}")
    print(f"figures in {figures}")


if __name__ == "__main__":
    main()
