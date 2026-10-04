"""Generate Figure 2 for preregistered E0--E2 and post-hoc E3--E5."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


import os
ROOT = Path(os.environ["EEG_RELEASE_ROOT"])


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def card(ax, x: float, y: float, w: float, h: float, label: str, title: str,
         lines: list[str], dimension: str, *, face: str, edge: str) -> None:
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.012",
        facecolor=face, edgecolor=edge, linewidth=0.8, transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(x + 0.018, y + h - 0.030, label, ha="left", va="top",
            fontsize=10.2, fontweight="bold", color=edge, transform=ax.transAxes)
    ax.text(x + 0.090, y + h - 0.029, title, ha="left", va="top",
            fontsize=7.35, fontweight="bold", color="#1f2937", transform=ax.transAxes)
    ax.text(x + 0.020, y + h - 0.075, "\n".join(f"• {line}" for line in lines),
            ha="left", va="top", fontsize=7.5, linespacing=1.28,
            color="#334155", transform=ax.transAxes)
    ax.text(x + 0.020, y + 0.018, dimension, ha="left", va="bottom",
            fontsize=7.8, fontweight="bold", color=edge, transform=ax.transAxes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    style()

    fig = plt.figure(figsize=(6.5, 4.65))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    navy = "#16324f"
    prereg_face = "#eef5fb"
    prereg_edge = "#1d4f7a"
    post_face = "#fff6e5"
    post_edge = "#a16207"

    ax.text(0.03, 0.965, "(a)", fontweight="bold", fontsize=9.2, color=navy,
            transform=ax.transAxes, va="top")
    ax.text(0.070, 0.965, "PREREGISTERED EEG LIBRARY", fontweight="bold", fontsize=9.2,
            color=navy, transform=ax.transAxes, va="top")
    ax.plot([0.03, 0.97], [0.935, 0.935], color=prereg_edge, lw=1.2,
            transform=ax.transAxes, clip_on=False)

    card(ax, 0.035, 0.535, 0.285, 0.345, "E0", "Compact spectrum", [
        "log broadband power",
        "relative theta · alpha · beta",
        "frontal / temporal / posterior",
    ], "12 EEG dimensions", face=prereg_face, edge=prereg_edge)
    card(ax, 0.3575, 0.535, 0.285, 0.345, "E1", "Expanded spectrum", [
        "retains E0",
        "channel spectral topography",
        "across-window variability",
        "regional spectral entropy",
        "training-only PCA",
    ], "35 EEG dimensions", face=prereg_face, edge=prereg_edge)
    card(ax, 0.680, 0.535, 0.285, 0.345, "E2", "Covariance geometry", [
        "retains E1",
        "theta / alpha / beta covariance",
        "CAR contrast space",
        "shrinkage covariance",
        "tangent-space coordinates",
        "training-only PCA",
    ], "47 EEG dimensions", face=prereg_face, edge=prereg_edge)

    ax.text(0.03, 0.455, "(b)", fontweight="bold", fontsize=9.2, color=navy,
            transform=ax.transAxes, va="top")
    ax.text(0.070, 0.455, "POST-HOC EXPLORATORY REPRESENTATIONS", fontweight="bold", fontsize=9.2,
            color=navy, transform=ax.transAxes, va="top")
    ax.plot([0.03, 0.97], [0.425, 0.425], color=post_edge, lw=1.2,
            transform=ax.transAxes, clip_on=False)

    card(ax, 0.035, 0.075, 0.285, 0.300, "E3", "Phase-lag connectivity", [
        "dwPLI²",
        "theta / alpha / beta",
        "91 channel pairs",
        "PCA",
    ], "9 EEG dimensions", face=post_face, edge=post_edge)
    card(ax, 0.3575, 0.075, 0.285, 0.300, "E4", "Entropy and Hjorth", [
        "permutation entropy",
        "Hjorth mobility",
        "Hjorth complexity",
        "three scalp regions",
        "no PCA",
    ], "9 EEG dimensions", face=post_face, edge=post_edge)
    card(ax, 0.680, 0.075, 0.285, 0.300, "E5", "Spectral states", [
        "theta / alpha / beta trajectories",
        "occupancy",
        "high-state duration",
        "switching rate",
        "PCA",
    ], "9 EEG dimensions", face=post_face, edge=post_edge)

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
