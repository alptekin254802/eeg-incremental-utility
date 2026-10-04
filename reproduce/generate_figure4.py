"""Generate Figure 4 from the saved preregistered master result table."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


import os
ROOT = Path(os.environ["EEG_RELEASE_ROOT"])
SOURCE = ROOT / "audit" / "preregistered_analysis" / "PREREGISTERED_RESULTS_MASTER_TABLE.csv"


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.4,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = pd.read_csv(SOURCE).set_index("analysis")
    rows = ["Primary adaptive M3", "Fixed E0", "Fixed E1", "Fixed E2"]
    labels = [
        "Adaptive M3\n(primary)",
        "E0\n(regional spectrum)",
        "E1\n(expanded spectrum)",
        "E2\n(covariance geometry)",
    ]
    columns = ["Delta_LL", "Delta_AUROC", "Delta_Brier"]
    panel_labels = ["Δ log loss", "Δ AUROC", "Δ Brier score"]
    xlims = [(-0.036, 0.012), (-0.060, 0.012), (-0.020, 0.005)]
    for row in rows:
        if row not in data.index:
            raise ValueError(f"Missing preregistered result row: {row}")
    expected = [-0.0288, 0.0045, -0.0122, -0.0154]
    if [round(float(data.loc[r, "Delta_LL"]), 4) for r in rows] != expected:
        raise ValueError("Fixed/adaptive delta values do not match the current source artifact")

    style()
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.60), sharey=True)
    y = list(range(len(rows)))[::-1]
    for idx, (ax, column, xlabel, xlim) in enumerate(zip(axes, columns, panel_labels, xlims)):
        vals = [float(data.loc[row, column]) for row in rows]
        ax.axhspan(2.55, 3.45, color="#eef5fb", zorder=0)
        for yi in y:
            ax.axhline(yi, color="#dbe3ec", lw=0.45, zorder=0)
        ax.axvline(0, color="#1f2937", lw=0.85, zorder=0)
        for row_idx, (yi, value) in enumerate(zip(y, vals)):
            primary_row = row_idx == 0
            ax.scatter([value], [yi], s=52 if primary_row else 38,
                       facecolors="#16324f" if primary_row else "white",
                       edgecolors="#16324f" if primary_row else "#64748b",
                       marker="o", linewidths=1.0, zorder=3)
            place_left = value > xlim[1] - 0.35 * (xlim[1] - xlim[0])
            offset = -5 if place_left else 5
            alignment = "right" if place_left else "left"
            ax.annotate(f"{value:+.4f}", (value, yi), xytext=(offset, 0),
                        textcoords="offset points", ha=alignment, va="center", fontsize=7.2,
                        color="#1f2937")
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.55, 3.55)
        ax.set_yticks(y)
        if idx == 0:
            ax.set_yticklabels(labels)
            ax.get_yticklabels()[0].set_fontweight("bold")
            ax.get_yticklabels()[0].set_color("#16324f")
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_xlabel(xlabel, fontsize=8.3, labelpad=5)
        ax.tick_params(axis="x", labelsize=7.4, length=3, pad=2)
        ax.tick_params(axis="y", labelsize=7.7, length=0, pad=6)
        ax.grid(axis="x", color="#cbd5e1", lw=0.45, alpha=0.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.text(0.0, 1.03, f"({chr(97 + idx)})", transform=ax.transAxes,
                fontweight="bold", fontsize=9.2, color="#16324f", va="bottom")

    fig.text(0.61, 0.035, "Positive values favor EEG augmentation",
             ha="center", va="bottom", fontsize=7.3, color="#475569")
    fig.subplots_adjust(left=0.205, right=0.985, bottom=0.23, top=0.88, wspace=0.28)
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
