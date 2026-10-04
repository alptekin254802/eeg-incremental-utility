"""Generate Figure 1 from saved cohort/QC artifacts only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch


import os
ROOT = Path(os.environ["EEG_RELEASE_ROOT"])
COHORT_QC = ROOT / "audit" / "cohort_screen"
PRIMARY = ROOT / "audit" / "preregistered_analysis"


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.1,
            "patch.linewidth": 0.65,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def box(ax, x: float, y: float, w: float, h: float, text: str, *,
        face: str = "#f6f8fb", edge: str = "#334155", lw: float = 0.8,
        fs: float = 8.0, ls: str = "-") -> None:
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.012",
        facecolor=face, edgecolor=edge, linewidth=lw, linestyle=ls,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color="#1f2937", fontsize=fs, transform=ax.transAxes)


def arrow(ax, x0: float, y0: float, x1: float, y1: float, *, ls: str = "-") -> None:
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0), xycoords=ax.transAxes,
        arrowprops={"arrowstyle": "-|>", "lw": 0.8, "color": "#475569",
                    "linestyle": ls, "mutation_scale": 9},
    )


def read_counts() -> tuple[int, int, int, int, int, int]:
    feasibility = pd.read_csv(COHORT_QC / "robots_final_feasibility.csv")
    row = feasibility.loc[
        feasibility["definition"].eq("EEG+Eye+Behavior_quality_pass")
    ]
    if len(row) != 1:
        raise ValueError("Expected one measurement-eligibility row")
    candidate = int(row.iloc[0]["denominator"])
    eligible = int(row.iloc[0]["total_participants"])

    qc = pd.read_csv(
        ROOT / "audit" / "stage1_blind_feature_extraction"
        / "EEG_PREPROCESSING_QC.csv"
    )
    primary_n = int(qc["preprocessing_status"].eq("PASS").sum())

    summary = json.loads((PRIMARY / "PRIMARY_ANALYSIS_SUMMARY.json").read_text(encoding="utf-8"))
    experimental = int(summary["class_counts"]["Experimental_Y1"])
    control = int(summary["class_counts"]["Control_Y0"])
    sensitivity_n = int(summary["diagnosis_sensitivity"]["n"])
    sensitivity_yes = int(summary["diagnosis_sensitivity"]["yes"])
    sensitivity_no = int(summary["diagnosis_sensitivity"]["no"])
    if (candidate, eligible, primary_n, experimental, control, sensitivity_n) != (
        126, 98, 96, 41, 55, 84
    ) or (sensitivity_yes, sensitivity_no) != (54, 30):
        raise ValueError("Cohort/QC counts do not match the current authoritative artifacts")
    return candidate, eligible, primary_n, experimental, control, sensitivity_n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Output PDF path; SVG is written beside it")
    args = parser.parse_args()
    candidate, eligible, primary_n, experimental, control, sensitivity_n = read_counts()
    style()

    fig = plt.figure(figsize=(6.5, 4.7))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    navy = "#16324f"
    blue = "#dbeafe"
    teal = "#dff3f0"
    amber = "#fff2cc"
    gray = "#f1f5f9"

    # (a) Cohort construction
    ax.text(0.025, 0.965, "(a)", fontweight="bold", fontsize=9.2, color=navy,
            transform=ax.transAxes, va="top")
    ax.text(0.065, 0.965, "Cohort construction", fontweight="bold", fontsize=8.5,
            color=navy, transform=ax.transAxes, va="top")
    box(ax, 0.035, 0.795, 0.255, 0.095,
        f"EEG + object gaze\n+ game-summary sessions\nN = {candidate}",
        face=blue, edge=navy, fs=8.0)
    box(ax, 0.035, 0.655, 0.255, 0.075,
        f"Measurement eligibility\nN = {eligible}", face=gray, fs=8.1)
    box(ax, 0.035, 0.515, 0.255, 0.075,
        f"EEG preprocessing/QC\nN = {primary_n}", face=gray, fs=8.1)
    box(ax, 0.035, 0.355, 0.255, 0.085,
        f"Final primary cohort\n{experimental} ADHD · {control} Control",
        face=teal, edge="#237a70", fs=8.0)
    arrow(ax, 0.1625, 0.795, 0.1625, 0.735)
    arrow(ax, 0.1625, 0.655, 0.1625, 0.595)
    arrow(ax, 0.1625, 0.515, 0.1625, 0.440)
    box(ax, 0.035, 0.165, 0.255, 0.095,
        "Diagnosis-status sensitivity\nN = 84\n54 yes / 30 no",
        face="#f8fafc", edge="#64748b", lw=0.8, fs=7.9, ls=(0, (3, 2)))
    ax.text(0.1625, 0.285, "not another primary cohort", ha="center", va="center",
            fontsize=7.0, color="#64748b", transform=ax.transAxes)

    # (b) Non-EEG ladder
    ax.text(0.335, 0.965, "(b)", fontweight="bold", fontsize=9.2, color=navy,
            transform=ax.transAxes, va="top")
    ax.text(0.375, 0.965, "Non-EEG\nmodality ladder", fontweight="bold", fontsize=8.5,
            color=navy, transform=ax.transAxes, va="top")
    box(ax, 0.345, 0.770, 0.235, 0.110, "M0: demographics\n2 predictors",
        face=blue, edge=navy, fs=8.0)
    box(ax, 0.345, 0.585, 0.235, 0.110, "M1: M0 + behavior\n5 predictors",
        face=blue, edge=navy, fs=8.0)
    box(ax, 0.345, 0.400, 0.235, 0.110, "M2: M1 + object gaze\n7 predictors",
        face=blue, edge=navy, fs=8.0)
    arrow(ax, 0.4625, 0.755, 0.4625, 0.710)
    arrow(ax, 0.4625, 0.570, 0.4625, 0.525)
    ax.text(0.4625, 0.36, "fixed non-EEG baseline", ha="center", va="center",
            fontsize=7.3, color="#475569", transform=ax.transAxes)

    # (c) EEG evaluation
    ax.text(0.655, 0.965, "(c)", fontweight="bold", fontsize=9.2, color=navy,
            transform=ax.transAxes, va="top")
    ax.text(0.695, 0.965, "EEG incremental-value\nevaluation", fontweight="bold", fontsize=8.0,
            color=navy, transform=ax.transAxes, va="top", linespacing=1.05)
    ax.text(0.815, 0.895, "PREREGISTERED PRIMARY", ha="center", va="center",
            fontsize=7.5, fontweight="bold", color="#237a70", transform=ax.transAxes)
    box(ax, 0.680, 0.785, 0.075, 0.070, "M2", face=gray, edge="#475569", fs=8.5)
    box(ax, 0.830, 0.775, 0.135, 0.090, "Adaptive M3\nM2 + EEG", face=teal,
        edge="#237a70", fs=8.0)
    arrow(ax, 0.755, 0.820, 0.830, 0.820)
    box(ax, 0.680, 0.630, 0.285, 0.090,
        "Training-only selection\nE0 / E1 / E2 and C",
        face=amber, edge="#9a6b00", fs=7.9)
    arrow(ax, 0.8225, 0.720, 0.8975, 0.775, ls="--")
    box(ax, 0.680, 0.495, 0.285, 0.075,
        "10 outer 5-fold repeats\n5-fold inner selection",
        face=gray, edge="#475569", fs=7.35)
    box(ax, 0.680, 0.390, 0.285, 0.060,
        "Held-out participants\nevaluation only",
        face="#f8fafc", edge="#64748b", lw=0.8, fs=7.7, ls=(0, (3, 2)))
    arrow(ax, 0.8225, 0.495, 0.8225, 0.450, ls="--")

    ax.text(0.815, 0.315, "POST-HOC BRANCH", ha="center", va="center",
            fontsize=7.5, fontweight="bold", color="#8a4b08", transform=ax.transAxes)
    box(ax, 0.680, 0.190, 0.085, 0.075, "M2 + E3", face=amber, edge="#a16207", fs=8.0)
    box(ax, 0.780, 0.190, 0.085, 0.075, "M2 + E4", face=amber, edge="#a16207", fs=8.0)
    box(ax, 0.880, 0.190, 0.085, 0.075, "M2 + E5", face=amber, edge="#a16207", fs=8.0)
    arrow(ax, 0.8225, 0.390, 0.7225, 0.265, ls="--")
    arrow(ax, 0.8225, 0.390, 0.8225, 0.265, ls="--")
    arrow(ax, 0.8225, 0.390, 0.9225, 0.265, ls="--")
    ax.text(0.815, 0.145, "separate exploratory comparisons\n(no E3/E4/E5 selection)",
            ha="center", va="center", fontsize=7.1, color="#64748b", transform=ax.transAxes,
            linespacing=1.05)

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
