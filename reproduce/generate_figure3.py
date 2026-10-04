"""Generate Figure 3 from the saved non-EEG modality-ladder table."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


import os
ROOT = Path(os.environ["EEG_RELEASE_ROOT"])
SOURCE = ROOT / "audit" / "preregistered_analysis" / "MODALITY_LADDER_CONSOLIDATION.csv"


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.15,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = pd.read_csv(SOURCE)
    absolute = data[data["row_type"].eq("absolute")].set_index("to_model")
    models = ["M0", "M1", "M2"]
    values = {
        "Log loss": [float(absolute.loc[m, "logloss"]) for m in models],
        "AUROC": [float(absolute.loc[m, "auroc"]) for m in models],
        "Brier score": [float(absolute.loc[m, "brier"]) for m in models],
    }
    if [round(v, 4) for v in values["Log loss"]] != [0.6461, 0.6256, 0.6244]:
        raise ValueError("Modality-ladder values do not match the current source artifact")

    style()
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.60), sharex=True)
    color = "#1d4f7a"
    x = [0, 1, 2]
    directions = ["lower is better", "higher is better", "lower is better"]
    limits = [(0.620, 0.650), (0.660, 0.725), (0.210, 0.230)]
    ticks = [
        [0.62, 0.63, 0.64, 0.65],
        [0.66, 0.68, 0.70, 0.72],
        [0.21, 0.22, 0.23],
    ]
    for idx, (ax, (metric, vals), direction, ylim, yticks) in enumerate(
        zip(axes, values.items(), directions, limits, ticks)
    ):
        ax.plot(x, vals, color=color, lw=1.25, marker="o", markersize=4.8,
                markerfacecolor="white", markeredgecolor=color,
                markeredgewidth=1.0, zorder=2)
        for xi, yi in zip(x, vals):
            ax.annotate(f"{yi:.3f}", (xi, yi), xytext=(0, 8), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.1, color="#334155")
        ax.set_xticks(x, ["M0", "M1", "M2"])
        ax.set_xlim(-0.22, 2.22)
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        ax.set_ylabel(metric, fontsize=8.2)
        ax.set_xlabel(direction, fontsize=7.2, color="#475569", labelpad=4)
        ax.tick_params(axis="both", labelsize=7.6, length=3, pad=2)
        ax.grid(axis="y", color="#cbd5e1", lw=0.45, alpha=0.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(-0.02, 1.03, f"({chr(97 + idx)})", transform=ax.transAxes,
                fontweight="bold", fontsize=9.2, color="#16324f", va="bottom")

    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.22, top=0.88, wspace=0.34)
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    pdf = out if out.suffix.lower() == ".pdf" else out.with_suffix(".pdf")
    svg = pdf.with_suffix(".svg")
    fig.savefig(pdf)
    fig.savefig(svg)
    plt.close(fig)


if __name__ == "__main__":
    main()
