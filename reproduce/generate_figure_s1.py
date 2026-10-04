"""Generate supplementary calibration diagnostics from saved calibration bins."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


import os
ROOT = Path(os.environ["EEG_RELEASE_ROOT"])
SOURCE = ROOT / "audit" / "preregistered_analysis" / "CALIBRATION_RESULTS.csv"


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.0,
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
    bins = data[data["record_type"].eq("reliability_bin")]
    scalar = data[data["record_type"].eq("scalar")]
    models = ["M2", "adaptive_M3"]
    failures = {
        model: set(scalar.loc[
            scalar["model"].eq(model) & scalar["failure_code"].notna(), "repeat"
        ].astype(int))
        for model in models
    }
    if failures["M2"] != {0, 1, 2, 4, 9} or failures["adaptive_M3"] != {0, 1, 4, 6, 7}:
        raise ValueError("Calibration failure status does not match the saved artifact")
    for model in models:
        if len(bins[bins["model"].eq(model)]) != 40:
            raise ValueError("Expected four saved reliability bins per repeat and model")

    style()
    fig, axes = plt.subplots(2, 5, figsize=(6.5, 4.65), sharex=True, sharey=True)
    colors = {"M2": "#1d4f7a", "adaptive_M3": "#a16207"}
    markers = {"M2": "o", "adaptive_M3": "s"}
    for repeat, ax in enumerate(axes.flat):
        ax.plot([0, 1], [0, 1], color="#64748b", lw=0.7, ls=(0, (3, 2)), zorder=0)
        for model in models:
            subset = bins[bins["model"].eq(model) & bins["repeat"].eq(repeat)].sort_values("bin")
            ax.plot(subset["mean_probability"], subset["observed_rate"], color=colors[model],
                    marker=markers[model], markersize=3.2, lw=0.9, label=model.replace("adaptive_", ""),
                    zorder=2)
        m2_status = "fail" if repeat in failures["M2"] else "fit"
        m3_status = "fail" if repeat in failures["adaptive_M3"] else "fit"
        ax.set_title(f"Repeat {repeat + 1}\nM2: {m2_status} · M3: {m3_status}", fontsize=7.2, pad=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1.0], ["0", ".5", "1"])
        ax.set_yticks([0, 0.5, 1.0], ["0", ".5", "1"])
        ax.tick_params(axis="both", labelsize=7.0, length=2, pad=1)
        ax.grid(color="#e2e8f0", lw=0.4, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if repeat == 0:
            ax.legend(fontsize=7.0, frameon=False, loc="lower right", handlelength=1.4,
                      borderpad=0.1, labelspacing=0.2)

    fig.text(0.5, 0.022, "Mean predicted probability", ha="center", va="bottom", fontsize=8.2)
    fig.text(0.008, 0.5, "Observed event rate", ha="center", va="center", rotation=90, fontsize=8.2)
    fig.text(0.5, 0.985, "Scalar-fit status: M2 failed in 5/10 repeats · adaptive M3 failed in 5/10 repeats",
             ha="center", va="top", fontsize=8.2, color="#1f2937")
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.10, top=0.875, wspace=0.27, hspace=0.48)
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
