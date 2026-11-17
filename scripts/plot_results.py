"""Figures and tables from the evaluation CSVs.

    python scripts/plot_results.py

Reads every ``results_and_figs/tables/{finetune,linear_probe}_s<seed>.csv`` (seeds are averaged;
bands show +-1 SD across seeds) and each run's ``metrics.csv`` in the checkpoints folder. Writes,
dated (PDF + PNG):

- ``data_efficiency_finetune``: test accuracy vs labelled examples, fine-tuning the whole network.
- ``data_efficiency_probe``: the same for a linear probe on the frozen last layer.
- ``layer_probe``: probe accuracy per layer, at a few labelled-set sizes.
- ``ablation_probe``: every LPL variant (pairing and temporal controls) side by side.
- ``temporal_rho``: probe accuracy vs clip smoothness rho (dose-response).
- ``training_progress``: the online probe (10k labels) over pre-training steps.

and the tables ``results_summary.csv`` (mean, SD, seeds per model / layer / n) and
``data_equivalence.csv`` (labels each prior is worth vs random init, on seed-mean curves).

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
from src.utils.paths import REPO_ROOT, out_root

STYLE = {  # model -> (label, colour, linestyle)
    "lpl": ("LPL", "#2a78d6", "-"),
    "simclr": ("SimCLR", "#eb6834", "-"),
    "oja": ("Oja", "#1baf7a", "-"),
    "random": ("Random init", "#52514e", "--"),
}
UPSTREAM = ("zenke_reference", "LPL, upstream code", "#2a78d6", ":")
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ORDINAL_BLUE = ("#86b6ef", "#2a78d6", "#104281")  # sequential blue, steps 250 / 450 / 650
LAST_LAYER = 8

# Ablation bars, top to bottom: (model, label, colour). LPL variants share LPL's blue.
ABLATIONS = [
    ("lpl_temporal_rho0p8", "Clips, smooth (ρ 0.8)", STYLE["lpl"][1]),
    ("lpl_temporal_ordershuf", "Clips, ρ 0.8, frame order shuffled", STYLE["lpl"][1]),
    ("lpl_temporal_rho0p5", "Clips, ρ 0.5", STYLE["lpl"][1]),
    ("lpl_temporal_rho0p95", "Clips, very smooth (ρ 0.95)", STYLE["lpl"][1]),
    ("lpl_temporal_rho0", "Clips, independent frames (ρ 0)", STYLE["lpl"][1]),
    ("lpl", "Two views (standard LPL)", STYLE["lpl"][1]),
    ("zenke_reference", "Two views, upstream code", STYLE["lpl"][1]),
    ("lpl_temporal_crossclip", "Clips, partner frame from another clip", STYLE["lpl"][1]),
    ("lpl_shuffled", "Two views, partner from another image", STYLE["lpl"][1]),
    ("oja", "Oja", STYLE["oja"][1]),
    ("random", "Random init", STYLE["random"][1]),
]
RHO_RUNS = {
    0.0: "lpl_temporal_rho0",
    0.5: "lpl_temporal_rho0p5",
    0.8: "lpl_temporal_rho0p8",
    0.95: "lpl_temporal_rho0p95",
}


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


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".pdf"), format="pdf")
    fig.savefig(path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def read_tables(tables: Path, kind: str, keys: list[str]) -> pd.DataFrame:
    """All seeds' rows for one protocol; the last row wins for a repeated (keys, seed)."""
    files = sorted(tables.glob(f"{kind}_s*.csv"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df.drop_duplicates([*keys, "seed"], keep="last")


def summarise(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Mean, SD (0 for one seed) and number of seeds of ``test_acc`` per ``keys``."""
    out = df.groupby(keys).test_acc.agg(["mean", "std", "count"]).reset_index()
    return out.fillna({"std": 0.0}).rename(columns={"count": "seeds"})


def plot_curves(ax: plt.Axes, df: pd.DataFrame, x: str, upstream: bool = False) -> None:
    """One line per model (seed mean, +-1 SD band), direct-labelled at its right end."""
    styles = list(STYLE.items())
    if upstream:
        styles.insert(1, (UPSTREAM[0], UPSTREAM[1:]))
    ends = []
    for model, (label, colour, style) in styles:
        rows = df[df.model == model].sort_values(x)
        if rows.empty:
            continue
        mean, sd = rows["mean"] * 100, rows["std"] * 100
        ax.plot(
            rows[x], mean, style, color=colour, linewidth=1.6, marker="o", markersize=4, label=label
        )
        if rows.seeds.max() > 1:
            ax.fill_between(rows[x], mean - sd, mean + sd, color=colour, alpha=0.15, linewidth=0)
        ends.append([label, rows[x].iloc[-1], mean.iloc[-1]])
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


def top_legend(ax: plt.Axes, ncol: int = 4) -> None:
    ax.legend(
        frameon=False,
        fontsize=8,
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        ncol=ncol,
        borderaxespad=0.2,
    )


def data_efficiency_figure(df: pd.DataFrame, title: str, path: Path, upstream: bool) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    plot_curves(ax, df, "labeled_n", upstream)
    ax.set_xscale("log")
    setup_axes(ax, "Labelled training examples (log scale)", "Test accuracy (%)")
    ax.set_title(title, fontsize=10, color=TEXT, loc="left", pad=22)
    top_legend(ax, ncol=5)
    ax.set_xlim(right=ax.get_xlim()[1] * 3.5)  # room for the direct labels
    fig.tight_layout()
    save(fig, path)


def layer_figure(probe: pd.DataFrame, counts: list[int], path: Path) -> None:
    fig, axes = plt.subplots(1, len(counts), figsize=(3.3 * len(counts), 3.5), sharey=True)
    for ax, n in zip(axes, counts, strict=True):
        plot_curves(ax, probe[probe.labeled_n == n], "layer")
        setup_axes(ax, "Layer", "Test accuracy (%)" if ax is axes[0] else "")
        ax.set_xticks(range(1, LAST_LAYER + 1))
        ax.set_title(f"{n:,} labels", fontsize=9, color=TEXT, loc="left")
        ax.set_xlim(0.5, LAST_LAYER + 2.4)  # room for the direct labels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="upper right", ncol=4)
    fig.suptitle(
        "Linear probe on each layer's features", fontsize=10, color=TEXT, x=0.01, ha="left"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, path)


def ablation_figure(last: pd.DataFrame, counts: list[int], path: Path) -> None:
    """Horizontal bars: every LPL variant and the baselines, one panel per label count."""
    rows = [a for a in ABLATIONS if a[0] in set(last.model)]
    fig, axes = plt.subplots(
        1, len(counts), figsize=(3.4 * len(counts) + 2.6, 0.32 * len(rows) + 1.4), sharey=True
    )
    axes = [axes] if len(counts) == 1 else list(axes)
    y = list(range(len(rows)))[::-1]
    for ax, n in zip(axes, counts, strict=True):
        at_n = last[last.labeled_n == n].set_index("model")
        means = [at_n.loc[m, "mean"] * 100 for m, _, _ in rows]
        sds = [at_n.loc[m, "std"] * 100 for m, _, _ in rows]
        ax.barh(
            y, means, height=0.72, color=[c for _, _, c in rows], edgecolor="white", linewidth=2
        )
        for yi, mean, sd, (m, _, _) in zip(y, means, sds, rows, strict=True):
            spread = at_n.loc[m, "seeds"] > 1
            if spread:
                ax.errorbar(mean, yi, xerr=sd, color=TEXT, linewidth=1, capsize=2)
            text_x = mean + (sd if spread else 0) + 0.8
            ax.text(text_x, yi, f"{mean:.1f}", va="center", fontsize=7.5, color=TEXT)
        setup_axes(ax, "Test accuracy (%)", "")
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{n:,} labels", fontsize=9, color=TEXT, loc="left")
        ax.set_xlim(0, max(m + s for m, s in zip(means, sds, strict=True)) * 1.14)
    axes[0].set_yticks(y, [label for _, label, _ in rows], fontsize=8, color=TEXT)
    fig.suptitle(
        "What LPL needs: linear probe on the last layer (bars: 3-seed runs show ±1 SD)",
        fontsize=10,
        color=TEXT,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, path)


def rho_figure(last: pd.DataFrame, counts: list[int], path: Path) -> None:
    """Probe accuracy vs clip smoothness rho, one line per label count."""
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ends = []
    for n, colour in zip(counts, ORDINAL_BLUE, strict=False):
        at_n = last[last.labeled_n == n].set_index("model")
        pts = [(rho, at_n.loc[m, "mean"] * 100) for rho, m in RHO_RUNS.items() if m in at_n.index]
        if not pts:
            continue
        xs, ys = zip(*pts, strict=True)
        ax.plot(
            xs,
            ys,
            "-",
            color=colour,
            linewidth=1.6,
            marker="o",
            markersize=5,
            label=f"{n:,} labels",
        )
        if "lpl_temporal_ordershuf" in at_n.index:
            ax.plot(
                0.8,
                at_n.loc["lpl_temporal_ordershuf", "mean"] * 100,
                "x",
                color=colour,
                markersize=8,
                markeredgewidth=1.6,
            )
        ends.append([f"{n:,} labels", xs[-1], ys[-1]])
    setup_axes(ax, "Clip smoothness ρ (frame-to-frame correlation)", "Test accuracy (%)")
    ax.set_xticks(list(RHO_RUNS))
    ax.set_xlim(-0.05, 1.12)
    label_ends(ax, ends)
    ax.plot(
        [],
        [],
        "x",
        color=MUTED,
        markersize=8,
        markeredgewidth=1.6,
        label="ρ 0.8, frame order shuffled",
    )
    ax.set_title(
        "LPL on synthetic clips: how smooth should time be?",
        fontsize=10,
        color=TEXT,
        loc="left",
        pad=34,
    )
    top_legend(ax, ncol=2)
    fig.tight_layout()
    save(fig, path)


def training_progress(checkpoints: Path) -> pd.DataFrame:
    """Online probe accuracy over pre-training steps, from every run's metrics.csv."""
    frames = []
    for model in ("lpl", "simclr", "oja"):
        for f in sorted(checkpoints.glob(f"{model}_s[0-9]/metrics.csv")):
            m = pd.read_csv(f)
            m = m[m.metric == "probe_acc"].rename(columns={"value": "test_acc"})
            frames.append(m.assign(model=model, seed=int(f.parent.name.rsplit("_s", 1)[1])))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def progress_figure(progress: pd.DataFrame, path: Path) -> None:
    curves = summarise(progress, ["model", "step"])
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    start = curves[curves.step == 0]["mean"].mean() * 100
    ax.axhline(
        start, color=STYLE["random"][1], linestyle="--", linewidth=1.4, label=STYLE["random"][0]
    )
    plot_curves(ax, curves[curves.model != "random"], "step")
    setup_axes(ax, "Pre-training step (2,048 frames each)", "Test accuracy (%)")
    ax.set_title(
        "Linear probe (10k labels) during pre-training", fontsize=10, color=TEXT, loc="left", pad=22
    )
    top_legend(ax)
    ax.set_xlim(right=ax.get_xlim()[1] * 1.15)
    fig.tight_layout()
    save(fig, path)


def equivalence_rows(df: pd.DataFrame, protocol: str) -> list[dict]:
    """Labels each model is worth vs random init, on seed-mean curves."""
    base = df[df.model == "random"].sort_values("labeled_n")
    rows = []
    for model in [m for m in df.model.unique() if m != "random"]:
        m = df[df.model == model].sort_values("labeled_n")
        if base.empty:
            continue
        for row in data_equivalence_table(
            m.labeled_n.tolist(), m["mean"].tolist(), base.labeled_n.tolist(), base["mean"].tolist()
        ):
            rows.append({"protocol": protocol, "model": model, **row})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--layer_counts", type=int, nargs="+", default=[100, 1000, 50000])
    parser.add_argument("--ablation_counts", type=int, nargs="+", default=[100, 1000, 50000])
    parser.add_argument("--tables", default=str(REPO_ROOT / "results_and_figs" / "tables"))
    parser.add_argument("--figures", default=str(REPO_ROOT / "results_and_figs" / "figures"))
    parser.add_argument("--checkpoints", default=str(out_root()))
    args = parser.parse_args()

    tables, figures = Path(args.tables), Path(args.figures)
    figures.mkdir(parents=True, exist_ok=True)
    stamp = f"{date.today():%Y%m%d}"
    summary, equivalence = [], []

    ft = read_tables(tables, "finetune", ["model", "labeled_n"])
    if not ft.empty:
        ft = summarise(ft, ["model", "labeled_n"])
        data_efficiency_figure(
            ft,
            "Fine-tuning the whole network",
            figures / f"data_efficiency_finetune_{stamp}",
            upstream=False,
        )
        summary.append(ft.assign(protocol="finetune", layer=LAST_LAYER))
        equivalence += equivalence_rows(ft, "finetune")

    probe = read_tables(tables, "linear_probe", ["model", "layer", "labeled_n"])
    if not probe.empty:
        probe = summarise(probe, ["model", "layer", "labeled_n"])
        last = probe[probe.layer == LAST_LAYER]
        data_efficiency_figure(
            last,
            "Linear probe on frozen last-layer features",
            figures / f"data_efficiency_probe_{stamp}",
            upstream=True,
        )
        layer_figure(probe, args.layer_counts, figures / f"layer_probe_{stamp}")
        ablation_figure(last, args.ablation_counts, figures / f"ablation_probe_{stamp}")
        rho_figure(last, args.ablation_counts, figures / f"temporal_rho_{stamp}")
        summary.append(probe.assign(protocol="linear_probe"))
        equivalence += equivalence_rows(last, "probe_last_layer")

    progress = training_progress(Path(args.checkpoints))
    if not progress.empty:
        progress_figure(progress, figures / f"training_progress_{stamp}")

    if summary:
        cols = ["protocol", "model", "layer", "labeled_n", "mean", "std", "seeds"]
        pd.concat(summary)[cols].to_csv(
            tables / "results_summary.csv", index=False, float_format="%.4f"
        )
    if equivalence:
        out = tables / "data_equivalence.csv"
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(equivalence[0]))
            writer.writeheader()
            writer.writerows(equivalence)
        print(f"wrote {out}")
    print(f"figures in {figures}")


if __name__ == "__main__":
    main()
