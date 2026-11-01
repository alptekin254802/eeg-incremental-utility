"""Create the exploratory master table from completed outputs only."""

from __future__ import annotations

import csv
import pandas as pd
import argparse
import json
from pathlib import Path

import numpy as np


OUT = Path(__file__).resolve().parent


def summary(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "iqr_type7": float(np.quantile(a, 0.75, method="linear") - np.quantile(a, 0.25, method="linear")),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    output=parser.parse_args().output.resolve()
    if output.is_relative_to(OUT.parents[1]): raise ValueError("Output must be outside the release")
    output.mkdir(parents=True, exist_ok=True)
    payload = json.loads((OUT / "BATCH_RESULTS_SUMMARY.json").read_text(encoding="utf-8"))
    rows = []
    for family in ("E3", "E4", "E5"):
        if family == "E5":
            repeat = pd.read_csv(OUT.parents[1] / "results/e5/REPEAT_RESULTS.csv").to_dict("records")
            entry = {"status": "COMPLETED", "repeat_rows": repeat}
        else:
            entry = payload["phase_b"]["families"][family]
        if entry["status"] != "COMPLETED":
            rows.append({"family": family, "status": entry["status"], "N": 96})
            continue
        repeat = entry["repeat_rows"]
        d_ll = summary([row["delta_ll"] for row in repeat])
        d_auc = summary([row["delta_auroc"] for row in repeat])
        d_brier = summary([row["delta_brier"] for row in repeat])
        rows.append({
            "family": family,
            "status": "COMPLETED",
            "N": 96,
            "model_without_EEG": "M2",
            "model_with_EEG": f"M2+{family}",
            "logloss_without_EEG": np.mean([row["m2_logloss"] for row in repeat]),
            "logloss_with_EEG": np.mean([row["eeg_logloss"] for row in repeat]),
            "Delta_LL": d_ll["mean"],
            "Delta_LL_median": d_ll["median"],
            "Delta_LL_IQR_type7": d_ll["iqr_type7"],
            "Delta_LL_min": d_ll["min"],
            "Delta_LL_max": d_ll["max"],
            "AUROC_without_EEG": np.mean([row["m2_auroc"] for row in repeat]),
            "AUROC_with_EEG": np.mean([row["eeg_auroc"] for row in repeat]),
            "Delta_AUROC": d_auc["mean"],
            "Brier_without_EEG": np.mean([row["m2_brier"] for row in repeat]),
            "Brier_with_EEG": np.mean([row["eeg_brier"] for row in repeat]),
            "Delta_Brier": d_brier["mean"],
            "interpretation": "Descriptive exploratory result; no family is designated a winner.",
        })
    fields = [
        "family", "status", "N", "model_without_EEG", "model_with_EEG",
        "logloss_without_EEG", "logloss_with_EEG", "Delta_LL", "Delta_LL_median",
        "Delta_LL_IQR_type7", "Delta_LL_min", "Delta_LL_max", "AUROC_without_EEG",
        "AUROC_with_EEG", "Delta_AUROC", "Brier_without_EEG", "Brier_with_EEG",
        "Delta_Brier", "interpretation",
    ]
    with (output / "EXPLORATORY_EEG_BATTERY_MASTER_TABLE.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
