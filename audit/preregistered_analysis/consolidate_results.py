"""Consolidate already-produced preregistered BALLADEER results.

This script intentionally reads only artifacts in this directory.  It does not
read the dataset, raw signals, frozen feature files, or fit any model.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent




def q_iqr(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    return float(np.quantile(values, 0.75, method="linear") - np.quantile(values, 0.25, method="linear"))


def summary(values: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "iqr": q_iqr(values),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def normalized_analysis(path: Path, analysis: str | None = None) -> pd.DataFrame:
    """Return common metric names without recomputing any predictions."""
    frame = pd.read_csv(path)
    if analysis is not None and "analysis" in frame.columns:
        frame = frame.loc[frame["analysis"].eq(analysis)].copy()

    def find_rep(prefix: str) -> str:
        candidates = [
            c for c in frame.columns
            if c.endswith(prefix) and c not in {f"m2_{prefix}", f"delta_{prefix}"}
        ]
        if len(candidates) != 1:
            raise ValueError(f"Could not identify representation {prefix} column in {path}: {candidates}")
        return candidates[0]

    out = pd.DataFrame({"repeat": frame["repeat"].astype(int)})
    for metric in ("logloss", "brier", "auroc"):
        rep_col = find_rep(metric)
        out[f"m2_{metric}"] = pd.to_numeric(frame[f"m2_{metric}"], errors="coerce")
        out[f"representation_{metric}"] = pd.to_numeric(frame[rep_col], errors="coerce")
        delta_col = "delta_ll" if metric == "logloss" else f"delta_{metric}"
        out[f"delta_{metric}"] = pd.to_numeric(frame[delta_col], errors="coerce")
    return out.sort_values("repeat").reset_index(drop=True)


def metric_summary(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "m2": {metric: summary(frame[f"m2_{metric}"]) for metric in ("logloss", "brier", "auroc")},
        "representation": {
            metric: summary(frame[f"representation_{metric}"]) for metric in ("logloss", "brier", "auroc")
        },
        "delta": {metric: summary(frame[f"delta_{metric}"]) for metric in ("logloss", "brier", "auroc")},
    }


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(OUTPUT / path.name, index=False, float_format="%.12g")


def main() -> None:
    global OUTPUT
    parser=argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    OUTPUT=parser.parse_args().output.resolve()
    if OUTPUT.is_relative_to(ROOT.parents[1]): raise ValueError("Output must be outside the release")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    primary = normalized_analysis(ROOT / "PRIMARY_REPEAT_SUMMARY.csv")
    e0 = normalized_analysis(ROOT / "E0_CORE_SENSITIVITY.csv")
    e1 = normalized_analysis(ROOT / "FIXED_E1_SECONDARY.csv")
    e2 = normalized_analysis(ROOT / "FIXED_E2_SECONDARY.csv")
    diagnosed = normalized_analysis(ROOT / "DIAGNOSED_SENSITIVITY.csv")


    analyses = {
        "Primary adaptive M3": (primary, "primary preregistered", 96, "M2", "Adaptive E0/E1/E2", "Primary estimate is negative; the adaptive EEG procedure has higher mean held-out log-loss than M2."),
        "Fixed E0": (e0, "required sensitivity", 96, "M2", "E0", "Required sensitivity only; not co-primary."),
        "Fixed E1": (e1, "secondary/descriptive", 96, "M2", "E1", "Secondary/descriptive fixed-representation result; it does not replace the primary adaptive estimate."),
        "Fixed E2": (e2, "secondary/descriptive", 96, "M2", "E2", "Secondary/descriptive fixed-representation result; it does not replace the primary adaptive estimate."),
        "Diagnosed sensitivity": (diagnosed, "non-primary sensitivity", 84, "M2", "Adaptive E0/E1/E2", "Non-primary diagnosis-status sensitivity; its log-loss direction is compared with, but does not alter, the primary group result."),
    }

    master_rows = []
    summaries: dict[str, dict[str, object]] = {}
    for name, (frame, status, n, baseline, rep, interpretation) in analyses.items():
        s = metric_summary(frame)
        summaries[name] = s
        master_rows.append({
            "analysis": name,
            "status": status,
            "N": n,
            "model_without_EEG": baseline,
            "EEG_representation": rep,
            "logloss_without_EEG": s["m2"]["logloss"]["mean"],
            "logloss_with_EEG": s["representation"]["logloss"]["mean"],
            "Delta_LL": s["delta"]["logloss"]["mean"],
            "AUROC_without_EEG": s["m2"]["auroc"]["mean"],
            "AUROC_with_EEG": s["representation"]["auroc"]["mean"],
            "Delta_AUROC": s["delta"]["auroc"]["mean"],
            "Brier_without_EEG": s["m2"]["brier"]["mean"],
            "Brier_with_EEG": s["representation"]["brier"]["mean"],
            "Delta_Brier": s["delta"]["brier"]["mean"],
            "interpretation": interpretation,
        })

    master = pd.DataFrame(master_rows)
    write_csv(ROOT / "PREREGISTERED_RESULTS_MASTER_TABLE.csv", master)

    ladder = pd.read_csv(ROOT / "EXPLORATORY_MODALITY_LADDER.csv")
    ladder_means = ladder.groupby("model", sort=False)[["logloss", "brier", "auroc"]].mean()
    ladder_rows = []
    for model in ("M0", "M1", "M2"):
        ladder_rows.append({
            "row_type": "absolute",
            "comparison": model,
            "from_model": "",
            "to_model": model,
            "N": 96,
            "logloss": ladder_means.loc[model, "logloss"],
            "brier": ladder_means.loc[model, "brier"],
            "auroc": ladder_means.loc[model, "auroc"],
            "delta_logloss_favoring_next": np.nan,
            "delta_brier_favoring_next": np.nan,
            "delta_auroc_favoring_next": np.nan,
        })
    for from_model, to_model in (("M0", "M1"), ("M1", "M2")):
        left = ladder.loc[ladder["model"].eq(from_model)].sort_values("repeat").reset_index(drop=True)
        right = ladder.loc[ladder["model"].eq(to_model)].sort_values("repeat").reset_index(drop=True)
        ladder_rows.append({
            "row_type": "increment",
            "comparison": f"{from_model} -> {to_model}",
            "from_model": from_model,
            "to_model": to_model,
            "N": 96,
            "logloss": np.nan,
            "brier": np.nan,
            "auroc": np.nan,
            "delta_logloss_favoring_next": float((left["logloss"] - right["logloss"]).mean()),
            "delta_brier_favoring_next": float((left["brier"] - right["brier"]).mean()),
            "delta_auroc_favoring_next": float((right["auroc"] - left["auroc"]).mean()),
        })
    write_csv(ROOT / "MODALITY_LADDER_CONSOLIDATION.csv", pd.DataFrame(ladder_rows))


if __name__ == "__main__":
    main()
