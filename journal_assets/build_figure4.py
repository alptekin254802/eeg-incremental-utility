"""Draw the device-transfer comparison from the complete saved result tables."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


def draw(results: Path, output: Path) -> dict:
    principal = pd.read_csv(results / "PRINCIPAL_CONTRASTS.csv", float_precision="round_trip")
    groups = pd.read_csv(results / "GROUP_CONTRASTS.csv", float_precision="round_trip")
    models = ["spectral_covariance_selector", "entropy_hjorth"]
    panels = [
        ("X_to_Plus", "a   EPOC X → EPOC+", "Target: 25 ADHD, 17 control", 25, 17),
        ("Plus_to_X", "b   EPOC+ → EPOC X", "Target: 16 ADHD, 38 control", 16, 38),
    ]
    styles = [("Control", "o", "#28648c"), ("ADHD", "s", "#bb563b"),
              ("Observed mean", "D", "#292929"), ("Equal-class mean", "X", "#138476")]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.55), sharex=True, sharey=True)
    fig.subplots_adjust(left=.245, right=.99, bottom=.35, top=.76, wspace=.16)
    audit = []
    for ax, (direction, title, subtitle, na, nc) in zip(axes, panels):
        ax.axvline(0, color="#858585", lw=.8, zorder=0)
        for y, model in enumerate(models):
            r = principal.loc[(principal.direction == direction) & (principal.model == model)]
            g = groups.loc[(groups.direction == direction) & (groups.model == model)].set_index("group")
            if len(r) != 1 or set(g.index) != {0, 1}:
                raise ValueError(f"Incomplete or duplicate comparison: {direction}, {model}")
            r = r.iloc[0]
            control, adhd = float(g.loc[0, "delta_ll"]), float(g.loc[1, "delta_ll"])
            values = [control, adhd, float(r.delta_ll), float(r.equal_class_delta_ll)]
            if not np.allclose(values[2:], [(nc * control + na * adhd) / (na + nc),
                                           (control + adhd) / 2], rtol=0, atol=1e-12):
                raise ValueError("Saved group-specific and aggregate increments disagree")
            for value, (_, marker, color) in zip(values, styles):
                ax.scatter(value, y, marker=marker, s=34, color=color, zorder=3,
                           linewidths=.5, edgecolors="white")
            audit.append(dict(direction=direction, model=model, control=control, adhd=adhd,
                              observed_mean=values[2], equal_class_mean=values[3]))
        ax.text(0, 1.24, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=10)
        ax.text(0, 1.03, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=9,
                color="#4b4b4b")
        ax.set_xlim(-.10, .16)
        ax.set_ylim(1.45, -.45)
        ax.set_xticks([-.10, 0, .10])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: "0" if abs(x) < 1e-10 else f"{x:+.2f}"))
        ax.grid(axis="x", color="#eeeeee", lw=.7)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#bbbbbb")
        ax.tick_params(axis="y", length=0, pad=8)
        ax.tick_params(axis="x", length=3, color="#999999", labelsize=9)
    axes[0].set_yticks([0, 1], ["Spectral/covariance\nselector", "Entropy/Hjorth"])
    fig.text(.615, .16, "Log-loss change (baseline − EEG-augmented)", ha="center", fontsize=9.5)
    handles = [Line2D([], [], marker=m, color=c, linestyle="None", markersize=5.5, label=l)
               for l, m, c in styles]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.53, -.005), ncol=4,
               frameon=False, fontsize=9, handletextpad=.4, columnspacing=1.25)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "device_transfer.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(output / "device_transfer.png", dpi=220)
    plt.close(fig)
    return {"points": audit, "source": "Saved device-transfer results; no fitting or resampling",
            "group_and_aggregate_identities_verified": True, "size_inches": [7.2, 2.55]}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(draw(args.results, args.output), indent=2))
