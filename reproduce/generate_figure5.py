"""Generate Figure 5 from saved repeat-specific log-loss differences."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


import os
ROOT = Path(os.environ["EEG_RELEASE_ROOT"])
PRIMARY = ROOT / "audit" / "preregistered_analysis"
EXPLORATORY = ROOT / "audit" / "exploratory_eeg_battery"


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

    primary = pd.read_csv(PRIMARY / "PRIMARY_REPEAT_SUMMARY.csv")
    exploratory = pd.read_csv(EXPLORATORY / "REPEAT_LEVEL_RESULTS.csv")
    e5 = pd.read_csv(ROOT / "results/e5/REPEAT_RESULTS.csv")
    exploratory = pd.concat([exploratory.loc[~exploratory.family.eq("E5")], e5], ignore_index=True)
    master = pd.read_csv(EXPLORATORY / "EXPLORATORY_EEG_BATTERY_MASTER_TABLE.csv").set_index("family")
    primary_master = pd.read_csv(PRIMARY / "PREREGISTERED_RESULTS_MASTER_TABLE.csv").set_index("analysis")

    series = {
        "Adaptive M3\n(primary)": primary["delta_ll"].to_numpy(),
        "E3\n(post-hoc)": exploratory.loc[exploratory["family"].eq("E3"), "delta_ll"].to_numpy(),
        "E4\n(post-hoc)": exploratory.loc[exploratory["family"].eq("E4"), "delta_ll"].to_numpy(),
        "E5\n(post-hoc)": exploratory.loc[exploratory["family"].eq("E5"), "delta_ll"].to_numpy(),
    }
    if any(len(values) != 10 for values in series.values()):
        raise ValueError("Expected ten saved repeat-specific estimates for each representation")
    means = {
        "Adaptive M3\n(primary)": float(primary_master.loc["Primary adaptive M3", "Delta_LL"]),
        "E3\n(post-hoc)": float(master.loc["E3", "Delta_LL"]),
        "E4\n(post-hoc)": float(master.loc["E4", "Delta_LL"]),
        "E5\n(post-hoc)": float(e5.delta_ll.mean()),
    }
    expected_means = [-0.0288, 0.0127, 0.0267, -0.0093]
    if [round(v, 4) for v in means.values()] != expected_means:
        raise ValueError("Repeat-series means do not match the current source artifacts")

    style()
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    colors = ["#16324f", "#2a7f78", "#a16207", "#7c3aed"]
    markers = ["o", "^", "s", "D"]
    legend_labels = [
        "Adaptive M3 (primary)",
        "E3 (connectivity)",
        "E4 (temporal dynamics)",
        "E5 (spectral states)",
    ]
    offsets = [-0.21, -0.07, 0.07, 0.21]
    x = list(range(1, 11))
    ax.axhline(0, color="#1f2937", lw=0.95, zorder=1)
    for idx, ((_, values), legend_label) in enumerate(zip(series.items(), legend_labels)):
        shifted_x = [repeat + offsets[idx] for repeat in x]
        ax.scatter(shifted_x, values, s=39 if idx == 0 else 34,
                   c=colors[idx], marker=markers[idx], edgecolors="#1f2937",
                   linewidths=0.5, label=legend_label, zorder=3)

    ax.set_xlim(0.55, 10.45)
    ax.set_ylim(-0.130, 0.085)
    ax.set_xticks(x)
    ax.set_yticks([-0.12, -0.08, -0.04, 0.00, 0.04, 0.08])
    ax.set_xlabel("Outer-CV repeat", fontsize=8.5)
    ax.set_ylabel("ΔLL", fontsize=8.5)
    ax.tick_params(axis="both", labelsize=7.6, length=3, pad=2)
    ax.grid(axis="both", color="#cbd5e1", lw=0.45, alpha=0.55, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.01, 0.93, "Favors EEG", transform=ax.transAxes,
            ha="left", va="top", fontsize=7.2, color="#475569")
    ax.text(0.01, 0.04, "Favors M2", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=7.2, color="#475569")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4,
              frameon=False, fontsize=7.2, handletextpad=0.35,
              columnspacing=0.75, borderaxespad=0.0)
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.18, top=0.82)
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
