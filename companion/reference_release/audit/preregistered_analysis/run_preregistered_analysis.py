"""Outcome linkage, preregistered group comparisons and diagnosis-status sensitivity."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
from scipy import optimize, special
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "audit" / "preregistered_analysis"
PRIMARY_MATRIX_PATH = ROOT / "audit" / "stage1_blind_feature_extraction" / "BLIND_PRIMARY_FEATURE_MATRIX.csv"
STAGE1_QC_PATH = ROOT / "audit" / "stage1_blind_feature_extraction" / "EEG_PREPROCESSING_QC.csv"
STAGE1_CODE_PATH = ROOT / "audit" / "stage1_blind_feature_extraction" / "run_stage1_extraction.py"
E1_TOPO_PATH = ROOT / "audit" / "stage1_5_enriched_eeg_feasibility" / "E1_TOPOGRAPHIC_SPECTRAL_RAW.csv"
E1_TEMP_PATH = ROOT / "audit" / "stage1_5_enriched_eeg_feasibility" / "E1_TEMPORAL_IQR_RAW.csv"
E1_ENTROPY_PATH = ROOT / "audit" / "stage1_5_enriched_eeg_feasibility" / "E1_REGIONAL_ENTROPY.csv"
E2_NPZ_PATH = ROOT / "audit" / "stage1_5_enriched_eeg_feasibility" / "E2_PARTICIPANT_COVARIANCES.npz"
E2_INDEX_PATH = ROOT / "audit" / "stage1_5_enriched_eeg_feasibility" / "E2_PARTICIPANT_COVARIANCE_INDEX.csv"
STAGE1_5_CODE_PATH = ROOT / "audit" / "stage1_5_enriched_eeg_feasibility" / "run_stage1_5_audit.py"
METADATA_PATH = ROOT / "dataset" / "users_demographics.json"

ROOT_SEED = 20260824125
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
REPRESENTATIONS = ["E0", "E1", "E2"]
REP_ORDER = {"E0": 0, "E1": 1, "E2": 2}
BANDS = ["theta", "alpha", "beta"]
M2_COLUMNS = [
    "age", "gender", "mean_work_speed", "total_omissions", "total_commissions",
    "gaze_location_entropy", "gaze_spatial_dispersion",
]
E0_COLUMNS = [
    "frontal_log_broadband", "temporal_log_broadband", "posterior_log_broadband",
    "frontal_rel_theta", "temporal_rel_theta", "posterior_rel_theta",
    "frontal_rel_alpha", "temporal_rel_alpha", "posterior_rel_alpha",
    "frontal_rel_beta", "temporal_rel_beta", "posterior_rel_beta",
]


class AnalysisStop(RuntimeError):
    pass


class CandidateFailure(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: Any) -> int:
    payload = "|".join(str(p) for p in (ROOT_SEED, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"participant_id": str})


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            clean: dict[str, Any] = {}
            for field in fields:
                value = row.get(field, "")
                if isinstance(value, np.generic):
                    value = value.item()
                if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                    value = ""
                clean[field] = value
            writer.writerow(clean)


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def finite_matrix(frame: pd.DataFrame, columns: list[str]) -> bool:
    return bool(np.all(np.isfinite(frame.loc[:, columns].to_numpy(dtype=np.float64))))


def sym(cov: np.ndarray) -> np.ndarray:
    return (cov + cov.T) / 2.0


def spd_log(cov: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(sym(cov))
    if np.any(values <= 0) or not np.all(np.isfinite(values)):
        raise CandidateFailure("E2_LOG_NONSPD")
    return (vectors * np.log(values)) @ vectors.T


def spd_exp(mat: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(sym(mat))
    if not np.all(np.isfinite(values)):
        raise CandidateFailure("E2_EXP_NONFINITE")
    result = sym((vectors * np.exp(values)) @ vectors.T)
    trace = float(np.trace(result))
    if trace <= 0 or not np.isfinite(trace):
        raise CandidateFailure("E2_EXP_NONPOSITIVE_TRACE")
    return result / trace


def expected_artifact_hashes() -> dict[Path, str]:
    return {
        PRIMARY_MATRIX_PATH: "6a7d363574e1274fff1850e6a925fe375a9f64316759befbaa6c26e946f40dc9",
        STAGE1_QC_PATH: "b9772d630f622ad90879ee7bae4918da8ca0308bd332f82d64324ce3183dc5a8",
        STAGE1_CODE_PATH: "3bcb4f06216a68ab80fafa14363a6abf2ea1c627b96c08df387dc8c711740ae7",
        E1_TOPO_PATH: "59ab48ea4cbb78ff1dd7af45c5cc1a9ff2aaa797a1607a15ede055549ce1383c",
        E1_TEMP_PATH: "cf2dc4d00ade9faa9b536ce847190a4e2b89931694e31cbd7d46124fb2c648b9",
        E1_ENTROPY_PATH: "66405052f0fe6329dbd72b98819c8fc8a9f3400e7303bf420b01a270e3431286",
        ROOT / "audit" / "stage1_5_enriched_eeg_feasibility" / "E0_VERIFICATION.csv": "5554db83b6574d19b3ef26ed0cc74e34a5b966622b5dd8f5c0097c5b7ce9fa7b",
        ROOT / "audit" / "stage1_5_enriched_eeg_feasibility" / "E2_COVARIANCE_QC.csv": "bb2b8aa23a6742f13c2f80e6148f548a56f7ab3905cca821ac6d011a9933af04",
        ROOT / "audit" / "stage1_5_enriched_eeg_feasibility" / "EEG_REPRESENTATION_RANK_QC.csv": "5365b166c8dcef879c1d41ad1202f93bfc2a0d358cc83cea8e534c0c2756bd84",
        ROOT / "audit" / "stage1_5_enriched_eeg_feasibility" / "STAGE1_5_REPRODUCIBILITY.json": "6628990d628c81757c10ecbc77125c790fe0fc216d51d0e592d3fd524b4ac3fe",
        ROOT / "audit" / "stage1_5_enriched_eeg_feasibility" / "STAGE1_5_SOFTWARE_PROVENANCE.json": "cf89cce29b9cac0b7be175c6aaaf624f940292ea4d97672ffbbf3e84c5b815e6",
        E2_NPZ_PATH: "48fedf64428886af195402da756ae6a39cfe53e10215de7b1205c424806feb0e",
        E2_INDEX_PATH: "7affd62a7ecce1a01f446b863f36a49e0687f19eea037b2ac8368a7042002b43",
        STAGE1_5_CODE_PATH: "d645ea4739f8054db12a556334f063236e27c77bb462c22dec515f6ff504a15b",
    }


def load_measurements() -> tuple[pd.DataFrame, dict[str, Any]]:
    expected = expected_artifact_hashes()
    actual: dict[str, str] = {}
    mismatches: list[str] = []
    for path, wanted in expected.items():
        if not path.exists():
            mismatches.append(f"missing:{path}")
            continue
        observed = sha256(path)
        actual[str(path)] = observed
        if observed != wanted:
            mismatches.append(f"hash:{path}:expected={wanted}:actual={observed}")
    if mismatches:
        raise AnalysisStop("HASH_VALIDATION_FAILED|" + "|".join(mismatches))

    primary = read_csv(PRIMARY_MATRIX_PATH)
    if len(primary) != 96 or primary["participant_id"].nunique() != 96 or primary["participant_id"].duplicated().any():
        raise AnalysisStop("PRIMARY_ROSTER_NOT_96_UNIQUE")
    ids = primary["participant_id"].astype(str).tolist()
    required = [*M2_COLUMNS, *E0_COLUMNS]
    if any(col not in primary.columns for col in required) or not finite_matrix(primary, required):
        raise AnalysisStop("PRIMARY_MATRIX_COLUMNS_OR_FINITE_FAILURE")

    topo = read_csv(E1_TOPO_PATH)
    temporal = read_csv(E1_TEMP_PATH)
    entropy = read_csv(E1_ENTROPY_PATH)
    blocks: dict[str, tuple[pd.DataFrame, int]] = {
        "topographic": (topo, 70), "temporal": (temporal, 15), "entropy": (entropy, 3),
    }
    for name, (frame, expected_n) in blocks.items():
        cols = [col for col in frame.columns if col != "participant_id"]
        if len(frame) != 96 or frame["participant_id"].nunique() != 96 or set(frame["participant_id"].astype(str)) != set(ids):
            raise AnalysisStop(f"E1_{name.upper()}_PARTICIPANT_SET_FAILURE")
        if len(cols) != expected_n or not finite_matrix(frame, cols):
            raise AnalysisStop(f"E1_{name.upper()}_FINITE_DIMENSION_FAILURE")
        frame = frame.set_index("participant_id").loc[ids].reset_index()
        blocks[name] = (frame, expected_n)

    with np.load(E2_NPZ_PATH, allow_pickle=False) as data:
        cov_ids = [str(x) for x in data["participant_id"].tolist()]
        bands = [str(x) for x in data["bands"].tolist()]
        channel_order = [str(x) for x in data["channel_order"].tolist()]
        covariances = np.asarray(data["covariances"], dtype=np.float64)
    wanted_channels = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
    if cov_ids != ids or bands != BANDS or channel_order != wanted_channels:
        raise AnalysisStop("E2_ID_BAND_CHANNEL_ORDER_FAILURE")
    if covariances.shape != (96, 3, 13, 13) or not np.all(np.isfinite(covariances)):
        raise AnalysisStop("E2_SHAPE_OR_FINITE_FAILURE")
    eig = np.linalg.eigvalsh(covariances.reshape(-1, 13, 13))
    if np.any(eig <= 0) or np.max(np.abs(covariances - np.swapaxes(covariances, -1, -2))) > 1e-12:
        raise AnalysisStop("E2_SPD_SYMMETRY_FAILURE")
    index = read_csv(E2_INDEX_PATH)
    if len(index) != 96 or index["participant_id"].astype(str).tolist() != ids:
        raise AnalysisStop("E2_INDEX_FAILURE")

    primary = primary.copy()
    primary["participant_id"] = ids
    for name, (frame, _) in blocks.items():
        for col in frame.columns:
            if col != "participant_id":
                primary[col] = frame[col].to_numpy(dtype=np.float64)
    measurement = {
        "primary": primary, "ids": ids,
        "topo_columns": [col for col in topo.columns if col != "participant_id"],
        "temporal_columns": [col for col in temporal.columns if col != "participant_id"],
        "entropy_columns": [col for col in entropy.columns if col != "participant_id"],
        "covariances": covariances, "hashes": actual,
        "e2_min_eigenvalue": float(eig.min()),
    }
    qc = read_csv(STAGE1_QC_PATH)
    if len(qc) != 98:
        raise AnalysisStop("STAGE1_QC_ROW_COUNT_FAILURE")
    return primary, measurement


def link_outcomes(primary: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if not isinstance(metadata, list):
        raise AnalysisStop("OFFICIAL_METADATA_NOT_LIST")
    raw_ids = [str(row.get("user", "")) for row in metadata]
    duplicates = sorted(pid for pid, count in Counter(raw_ids).items() if pid and count > 1)
    if duplicates:
        raise AnalysisStop("DUPLICATE_METADATA_IDS|" + ",".join(duplicates))
    by_id = {str(row.get("user")): row for row in metadata if str(row.get("user", ""))}
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    bad_group: list[str] = []
    for pid in primary["participant_id"].astype(str):
        row = by_id.get(pid)
        if row is None:
            missing.append(pid)
            continue
        try:
            group = int(row.get("group"))
        except Exception:
            group = None
        if group not in (1, 2):
            bad_group.append(pid)
        diagnosed = str(row.get("diagnosed", "")).strip().lower()
        rows.append({
            "participant_id": pid, "group_raw": row.get("group", ""),
            "group_y": 1 if group == 1 else (0 if group == 2 else ""),
            "diagnosed_raw": row.get("diagnosed", ""),
            "diagnosed_y": 1 if diagnosed == "yes" else (0 if diagnosed == "no" else ""),
            "age": row.get("age", ""), "gender": row.get("gender", ""),
            "metadata_source": "dataset/users_demographics.json",
        })
    if missing:
        raise AnalysisStop("PRIMARY_METADATA_MATCH_MISSING|" + ",".join(missing))
    if bad_group:
        raise AnalysisStop("PRIMARY_GROUP_UNRECOGNIZED|" + ",".join(bad_group))
    linked = pd.DataFrame(rows)
    if len(linked) != 96 or linked["participant_id"].nunique() != 96 or linked["group_y"].isna().any():
        raise AnalysisStop("OUTCOME_LINKAGE_NOT_EXACTLY_ONE_VALID_PRIMARY_ROW")
    group_counts = {str(k): int(v) for k, v in linked["group_y"].astype(int).value_counts().sort_index().items()}
    diagnosis = linked["diagnosed_raw"].astype(str).str.lower()
    summary = {
        "metadata_rows": len(metadata), "metadata_unique_ids": len(by_id),
        "extra_nonroster_metadata_ids": len(set(by_id) - set(primary["participant_id"])),
        "roster_rows": 96, "matched_rows": len(linked), "primary_group_counts": group_counts,
        "diagnosed_yes": int((diagnosis == "yes").sum()), "diagnosed_no": int((diagnosis == "no").sum()),
        "diagnosed_undetermined": int((diagnosis == "undetermined").sum()),
        "diagnosed_other_or_missing": int((~diagnosis.isin(["yes", "no", "undetermined"])).sum()),
    }
    return linked, summary




def build_folds(ids: list[str], y: np.ndarray, outcome: str) -> dict[str, Any]:
    outer_records: list[dict[str, Any]] = []
    seed_manifest: dict[str, Any] = {
        "outcome": outcome, "root_seed": ROOT_SEED,
        "stable_seed": "int.from_bytes(hashlib.sha256('|'.join(str(p) for p in (20260824125,*parts)).encode('utf-8')).digest()[:4], 'little')",
        "repeats": [],
    }
    for repeat in range(10):
        outer_seed = stable_seed(outcome, "outer", repeat)
        outer_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=outer_seed)
        repeat_entry = {"repeat": repeat, "outer_seed": outer_seed, "outer_folds": []}
        for outer_fold, (train_local, test_local) in enumerate(outer_splitter.split(np.zeros(len(y)), y)):
            train = np.asarray(train_local, dtype=int)
            test = np.asarray(test_local, dtype=int)
            inner_seed = stable_seed(outcome, "inner", repeat, outer_fold)
            inner_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=inner_seed)
            inner_folds: list[dict[str, Any]] = []
            for inner_fold, (itr_local, iva_local) in enumerate(inner_splitter.split(np.zeros(len(train)), y[train])):
                inner_folds.append({
                    "inner_fold": inner_fold,
                    "inner_train": train[np.asarray(itr_local, dtype=int)],
                    "inner_valid": train[np.asarray(iva_local, dtype=int)],
                })
            outer_records.append({
                "repeat": repeat, "outer_fold": outer_fold, "outer_seed": outer_seed, "inner_seed": inner_seed,
                "outer_train": train, "outer_test": test, "inner_folds": inner_folds,
            })
            repeat_entry["outer_folds"].append({
                "outer_fold": outer_fold, "inner_seed": inner_seed,
                "outer_train_indices": train.tolist(), "outer_test_indices": test.tolist(),
                "inner_folds": [
                    {"inner_fold": item["inner_fold"], "inner_train_indices": item["inner_train"].tolist(), "inner_valid_indices": item["inner_valid"].tolist()}
                    for item in inner_folds
                ],
            })
        seed_manifest["repeats"].append(repeat_entry)
    return {"outcome": outcome, "ids": ids, "y": np.asarray(y, dtype=int), "outer": outer_records, "seed_manifest": seed_manifest}


def write_folds(primary: dict[str, Any], diagnosis: dict[str, Any] | None, hashes: dict[str, str]) -> None:
    primary_rows: list[dict[str, Any]] = []
    diagnosis_rows: list[dict[str, Any]] = []
    for folds, name in ((primary, "primary_group"), (diagnosis, "diagnosed_sensitivity")):
        if folds is None:
            continue
        rows = primary_rows if name == "primary_group" else diagnosis_rows
        for outer in folds["outer"]:
            train = set(outer["outer_train"].tolist())
            test = set(outer["outer_test"].tolist())
            for idx, pid in enumerate(folds["ids"]):
                if idx in train or idx in test:
                    rows.append({
                        "outcome": name, "level": "outer", "repeat": outer["repeat"], "outer_fold": outer["outer_fold"],
                        "inner_fold": "", "participant_id": pid, "index": idx,
                        "role": "train" if idx in train else "test", "seed": outer["outer_seed"],
                        "outer_seed": outer["outer_seed"], "inner_seed": outer["inner_seed"],
                    })
            for inner in outer["inner_folds"]:
                itr = set(inner["inner_train"].tolist())
                iva = set(inner["inner_valid"].tolist())
                for idx, pid in enumerate(folds["ids"]):
                    if idx in itr or idx in iva:
                        rows.append({
                            "outcome": name, "level": "inner", "repeat": outer["repeat"], "outer_fold": outer["outer_fold"],
                            "inner_fold": inner["inner_fold"], "participant_id": pid, "index": idx,
                            "role": "train" if idx in itr else "valid", "seed": outer["inner_seed"],
                            "outer_seed": outer["outer_seed"], "inner_seed": outer["inner_seed"],
                        })
    fields = [
        "outcome", "level", "repeat", "outer_fold", "inner_fold", "participant_id", "index", "role", "seed", "outer_seed", "inner_seed",
    ]
    write_csv(OUT / "PRIMARY_FOLD_MANIFEST.csv", primary_rows, fields)
    if diagnosis_rows:
        write_csv(OUT / "DIAGNOSED_FOLD_MANIFEST.csv", diagnosis_rows, fields)
    write_json(OUT / "PRIMARY_SEED_MANIFEST.json", {
        "root_seed": ROOT_SEED,
        "stable_seed_function": "int.from_bytes(hashlib.sha256('|'.join(str(p) for p in (20260824125,*parts)).encode('utf-8')).digest()[:4], 'little')",
        "primary_group": primary["seed_manifest"],
        "diagnosed_sensitivity": diagnosis["seed_manifest"] if diagnosis is not None else None,
        "primary_roster_sha256": sha256(PRIMARY_MATRIX_PATH),
        "measurement_hashes": hashes,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": __import__("sklearn").__version__},
    })


def prepare_raw(primary: pd.DataFrame, measurement: dict[str, Any], indices: list[int] | None = None) -> dict[str, np.ndarray]:
    frame = primary if indices is None else primary.iloc[indices].reset_index(drop=True)
    covariances = measurement["covariances"] if indices is None else measurement["covariances"][indices]
    return {
        "m0": frame[["age", "gender"]].to_numpy(dtype=np.float64),
        "m1": frame[["age", "gender", "mean_work_speed", "total_omissions", "total_commissions"]].to_numpy(dtype=np.float64),
        "m2": frame[M2_COLUMNS].to_numpy(dtype=np.float64),
        "e0": frame[E0_COLUMNS].to_numpy(dtype=np.float64),
        "topo": frame[measurement["topo_columns"]].to_numpy(dtype=np.float64),
        "temporal": frame[measurement["temporal_columns"]].to_numpy(dtype=np.float64),
        "entropy": frame[measurement["entropy_columns"]].to_numpy(dtype=np.float64),
        "covariances": covariances.astype(np.float64),
    }


def impute_scale_block(train: np.ndarray, assessment: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train = np.asarray(train, dtype=np.float64).copy()
    assessment = np.asarray(assessment, dtype=np.float64).copy()
    missing_rate = np.mean(~np.isfinite(train), axis=0)
    if np.any(missing_rate > 0.10):
        raise CandidateFailure(f"{name}_MISSINGNESS_GT_10_PERCENT")
    if np.any(~np.isfinite(train)) or np.any(~np.isfinite(assessment)):
        medians = np.nanmedian(train, axis=0)
        if np.any(~np.isfinite(medians)):
            raise CandidateFailure(f"{name}_NONFINITE_MEDIAN")
        for col in range(train.shape[1]):
            train[~np.isfinite(train[:, col]), col] = medians[col]
            assessment[~np.isfinite(assessment[:, col]), col] = medians[col]
    scaler = StandardScaler(with_mean=True, with_std=True)
    train_out = scaler.fit_transform(train).astype(np.float64)
    assessment_out = scaler.transform(assessment).astype(np.float64)
    if not np.all(np.isfinite(train_out)) or not np.all(np.isfinite(assessment_out)):
        raise CandidateFailure(f"{name}_NONFINITE_AFTER_SCALE")
    return train_out, assessment_out, {
        "missing_rate": missing_rate.tolist(), "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
    }


def final_scale(train: np.ndarray, assessment: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    scaler = StandardScaler(with_mean=True, with_std=True)
    train_out = scaler.fit_transform(np.asarray(train, dtype=np.float64)).astype(np.float64)
    assessment_out = scaler.transform(np.asarray(assessment, dtype=np.float64)).astype(np.float64)
    if not np.all(np.isfinite(train_out)) or not np.all(np.isfinite(assessment_out)):
        raise CandidateFailure(f"{name}_FINAL_SCALE_NONFINITE")
    return train_out, assessment_out, {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}


def tangent_coordinates(train_cov: np.ndarray, assessment_cov: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    reference = spd_exp(np.mean(np.stack([spd_log(cov) for cov in train_cov]), axis=0))
    values, vectors = np.linalg.eigh(sym(reference))
    if np.any(values <= 0) or not np.all(np.isfinite(values)):
        raise CandidateFailure("E2_REFERENCE_NONSPD")
    inv_sqrt = sym((vectors * (1.0 / np.sqrt(values))) @ vectors.T)
    tri = np.tril_indices(13)
    lower = tri[0] > tri[1]

    def transform(covariances: np.ndarray) -> np.ndarray:
        rows: list[np.ndarray] = []
        for cov in covariances:
            mapped = sym(inv_sqrt @ cov @ inv_sqrt)
            vector = spd_log(mapped)[tri].copy()
            vector[lower] *= math.sqrt(2.0)
            rows.append(vector)
        result = np.asarray(rows, dtype=np.float64)
        if not np.all(np.isfinite(result)):
            raise CandidateFailure("E2_TANGENT_NONFINITE")
        return result

    return transform(train_cov), transform(assessment_cov), {
        "reference_trace": float(np.trace(reference)),
        "reference_min_eigenvalue": float(np.linalg.eigvalsh(reference).min()),
        "coordinate_dimension": 91,
    }


def representation_features(
    raw: dict[str, np.ndarray],
    train_idx: np.ndarray,
    assessment_idx: np.ndarray,
    representation: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if representation == "M0":
        return final_scale(raw["m0"][train_idx], raw["m0"][assessment_idx], "M0")
    if representation == "M1":
        return final_scale(raw["m1"][train_idx], raw["m1"][assessment_idx], "M1")
    if representation == "M2":
        return final_scale(raw["m2"][train_idx], raw["m2"][assessment_idx], "M2")
    if representation not in REPRESENTATIONS:
        raise CandidateFailure("UNKNOWN_REPRESENTATION")

    m2_train = raw["m2"][train_idx]
    m2_assessment = raw["m2"][assessment_idx]
    e0_train = raw["e0"][train_idx]
    e0_assessment = raw["e0"][assessment_idx]
    if representation == "E0":
        return final_scale(
            np.column_stack([m2_train, e0_train]),
            np.column_stack([m2_assessment, e0_assessment]),
            "E0",
        )

    topo_train, topo_assessment, topo_meta = impute_scale_block(raw["topo"][train_idx], raw["topo"][assessment_idx], "E1_TOPO_INPUT")
    temporal_train, temporal_assessment, temporal_meta = impute_scale_block(raw["temporal"][train_idx], raw["temporal"][assessment_idx], "E1_TEMP_INPUT")
    if topo_train.shape[0] <= 15 or temporal_train.shape[0] <= 5:
        raise CandidateFailure("E1_PCA_SAMPLE_SIZE")
    topo_pca = PCA(n_components=15, svd_solver="full", whiten=False)
    temporal_pca = PCA(n_components=5, svd_solver="full", whiten=False)
    topo_train_pc = topo_pca.fit_transform(topo_train).astype(np.float64)
    topo_assessment_pc = topo_pca.transform(topo_assessment).astype(np.float64)
    temporal_train_pc = temporal_pca.fit_transform(temporal_train).astype(np.float64)
    temporal_assessment_pc = temporal_pca.transform(temporal_assessment).astype(np.float64)

    topo_rank = int(np.sum(topo_pca.singular_values_ > max(topo_train.shape) * np.finfo(float).eps * topo_pca.singular_values_[0]))
    temporal_rank = int(np.sum(temporal_pca.singular_values_ > max(temporal_train.shape) * np.finfo(float).eps * temporal_pca.singular_values_[0]))
    if topo_rank < 15 or temporal_rank < 5:
        raise CandidateFailure("E1_RANK_FAILURE")
    e1_train = np.column_stack([m2_train, e0_train, raw["entropy"][train_idx], topo_train_pc, temporal_train_pc])
    e1_assessment = np.column_stack([m2_assessment, e0_assessment, raw["entropy"][assessment_idx], topo_assessment_pc, temporal_assessment_pc])
    metadata: dict[str, Any] = {
        "topographic_rank": topo_rank, "temporal_rank": temporal_rank,
        "topographic_pca_singular_values": topo_pca.singular_values_.tolist(),
        "temporal_pca_singular_values": temporal_pca.singular_values_.tolist(),
        "topographic_input": topo_meta, "temporal_input": temporal_meta,
    }
    if representation == "E1":
        train_out, assessment_out, scale_meta = final_scale(e1_train, e1_assessment, "E1")
        metadata["final_scaler"] = scale_meta
        return train_out, assessment_out, metadata

    e2_train_parts: list[np.ndarray] = []
    e2_assessment_parts: list[np.ndarray] = []
    e2_metadata: dict[str, Any] = {}
    for band_i, band in enumerate(BANDS):
        train_coords, assessment_coords, ref_meta = tangent_coordinates(
            raw["covariances"][train_idx, band_i],
            raw["covariances"][assessment_idx, band_i],
        )
        pca = PCA(n_components=4, svd_solver="full", whiten=False)
        train_pc = pca.fit_transform(train_coords).astype(np.float64)
        assessment_pc = pca.transform(assessment_coords).astype(np.float64)
        rank = int(np.sum(pca.singular_values_ > max(train_coords.shape) * np.finfo(float).eps * pca.singular_values_[0]))
        if rank < 4:
            raise CandidateFailure(f"E2_{band}_RANK_FAILURE")
        e2_train_parts.append(train_pc)
        e2_assessment_parts.append(assessment_pc)
        e2_metadata[band] = {
            **ref_meta, "rank": rank, "pca_singular_values": pca.singular_values_.tolist(),
        }
    e2_train = np.column_stack(e2_train_parts)
    e2_assessment = np.column_stack(e2_assessment_parts)
    train_out, assessment_out, scale_meta = final_scale(
        np.column_stack([e1_train, e2_train]),
        np.column_stack([e1_assessment, e2_assessment]),
        "E2",
    )
    metadata["e2"] = e2_metadata
    metadata["final_scaler"] = scale_meta
    return train_out, assessment_out, metadata


def fit_logistic(
    train_x: np.ndarray,
    train_y: np.ndarray,
    assessment_x: np.ndarray,
    c_value: float,
    context: dict[str, Any],
    failures: list[dict[str, Any]],
) -> np.ndarray:
    convergence_seen = False
    for max_iter in (1000, 10000):
        try:
            model = LogisticRegression(
                penalty="l2", solver="lbfgs", C=float(c_value), tol=1e-6,
                max_iter=max_iter, fit_intercept=True, class_weight=None,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(np.asarray(train_x, dtype=np.float64), np.asarray(train_y, dtype=int))
            convergence = any(issubclass(item.category, ConvergenceWarning) for item in caught)
            if convergence:
                convergence_seen = True
                failures.append({**context, "stage": "logistic_fit", "error_code": "CONVERGENCE_WARNING_RETRY", "message": f"max_iter={max_iter}"})
                if max_iter == 1000:
                    continue
                raise CandidateFailure("CONVERGENCE_WARNING_AFTER_RETRY")
            if not np.all(np.isfinite(model.coef_)) or not np.all(np.isfinite(model.intercept_)):
                raise CandidateFailure("NONFINITE_LOGISTIC_COEFFICIENT")
            probability = model.predict_proba(np.asarray(assessment_x, dtype=np.float64))[:, 1]
            if not np.all(np.isfinite(probability)):
                raise CandidateFailure("NONFINITE_LOGISTIC_PROBABILITY")
            return probability
        except CandidateFailure as exc:
            failures.append({**context, "stage": "logistic_fit", "error_code": str(exc), "message": str(exc)})
            if max_iter == 1000 and not convergence_seen:
                raise
            if max_iter == 10000:
                raise
        except Exception as exc:
            failures.append({**context, "stage": "logistic_fit", "error_code": "LOGISTIC_EXCEPTION", "message": repr(exc)})
            raise CandidateFailure("LOGISTIC_EXCEPTION") from exc
    raise CandidateFailure("LOGISTIC_FAILURE")


def clipped_log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(p, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    outcome = np.asarray(y, dtype=np.float64)
    return -(outcome * np.log(probability) + (1.0 - outcome) * np.log(1.0 - probability))


def add_selection_row(
    rows: list[dict[str, Any]], folds: dict[str, Any], family: str, outer: dict[str, Any],
    representation: str, c_value: float, score: float, valid: bool, rank: int | str,
    selected_initial: bool, fallback_rank: int | str = "",
) -> None:
    rows.append({
        "outcome": folds["outcome"], "model_family": family, "repeat": outer["repeat"],
        "outer_fold": outer["outer_fold"], "representation": representation, "C": c_value,
        "pooled_logloss": score if valid else "", "rounded_pooled_logloss": round(score, 12) if valid else "",
        "valid": valid, "rank": rank, "selected_initial": selected_initial, "fallback_rank": fallback_rank,
    })


def run_tuned_model(
    folds: dict[str, Any],
    raw: dict[str, np.ndarray],
    mode: str,
    family: str,
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    if mode == "adaptive":
        candidates = [(rep, c) for rep in REPRESENTATIONS for c in C_GRID]
    else:
        candidates = [(mode, c) for c in C_GRID]
    predictions: list[dict[str, Any]] = []
    selection: list[dict[str, Any]] = []
    for outer in folds["outer"]:
        scores: dict[tuple[str, float], float] = {}
        valid: dict[tuple[str, float], bool] = {candidate: True for candidate in candidates}
        for rep, c_value in candidates:
            losses: list[float] = []
            for inner in outer["inner_folds"]:
                context = {
                    "outcome": folds["outcome"], "model_family": family, "repeat": outer["repeat"],
                    "outer_fold": outer["outer_fold"], "inner_fold": inner["inner_fold"],
                    "representation": rep, "C": c_value,
                }
                try:
                    train_x, valid_x, _ = representation_features(raw, inner["inner_train"], inner["inner_valid"], rep)
                    probability = fit_logistic(train_x, folds["y"][inner["inner_train"]], valid_x, c_value, context, failures)
                    losses.extend(clipped_log_loss(folds["y"][inner["inner_valid"]], probability).tolist())
                except CandidateFailure as exc:
                    valid[(rep, c_value)] = False
                    failures.append({**context, "stage": "inner_candidate", "error_code": str(exc), "message": str(exc)})
                    break
            scores[(rep, c_value)] = float(np.sum(losses) / len(outer["outer_train"])) if valid[(rep, c_value)] else float("inf")
        ranked = sorted(
            [(rep, c, scores[(rep, c)]) for rep, c in candidates if valid[(rep, c)]],
            key=lambda item: (round(item[2], 12), REP_ORDER.get(item[0], 99), float(item[1])),
        )
        if not ranked:
            raise AnalysisStop(f"NO_VALID_CANDIDATE|{family}|repeat={outer['repeat']}|outer={outer['outer_fold']}")
        initial = ranked[0]
        for rank, (rep, c, score) in enumerate(ranked, start=1):
            add_selection_row(selection, folds, family, outer, rep, c, score, True, rank, rank == 1)
        for rep, c in candidates:
            if not valid[(rep, c)]:
                add_selection_row(selection, folds, family, outer, rep, c, float("inf"), False, "", False)

        selected: tuple[str, float, float, int, np.ndarray] | None = None
        transform_cache: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
        for rank, (rep, c, score) in enumerate(ranked, start=1):
            context = {
                "outcome": folds["outcome"], "model_family": family, "repeat": outer["repeat"],
                "outer_fold": outer["outer_fold"], "inner_fold": "", "representation": rep, "C": c,
            }
            try:
                if rep not in transform_cache:
                    transform_cache[rep] = representation_features(raw, outer["outer_train"], outer["outer_test"], rep)
                train_x, test_x, _ = transform_cache[rep]
                probability = fit_logistic(train_x, folds["y"][outer["outer_train"]], test_x, c, context, failures)
                selected = (rep, c, score, rank, probability)
                break
            except CandidateFailure as exc:
                failures.append({**context, "stage": "outer_refit_fallback", "error_code": str(exc), "message": f"rank={rank}; next pre-ranked candidate"})
        if selected is None:
            raise AnalysisStop(f"NO_OUTER_REFIT|{family}|repeat={outer['repeat']}|outer={outer['outer_fold']}")
        rep, c, score, rank, probability = selected
        selection.append({
            "outcome": folds["outcome"], "model_family": f"{family}_SELECTED", "repeat": outer["repeat"],
            "outer_fold": outer["outer_fold"], "representation": rep, "C": c,
            "pooled_logloss": score, "rounded_pooled_logloss": round(score, 12), "valid": True,
            "rank": rank, "selected_initial": False, "fallback_rank": rank if rank != 1 else "",
        })
        for idx, p in zip(outer["outer_test"], probability):
            predictions.append({
                "outcome": folds["outcome"], "model_family": family, "repeat": outer["repeat"],
                "outer_fold": outer["outer_fold"], "participant_index": int(idx), "participant_id": folds["ids"][int(idx)],
                "y": int(folds["y"][int(idx)]), "probability": float(p), "representation": rep, "C": c,
                "initial_representation": initial[0], "initial_C": initial[1], "initial_inner_logloss": initial[2],
                "actual_inner_rank": rank,
            })
    return {"predictions": predictions, "selection": selection}


def prediction_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows).sort_values(["repeat", "participant_index"]).reset_index(drop=True)


def metric_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "iqr_type7": float(np.quantile(values, 0.75, method="linear") - np.quantile(values, 0.25, method="linear")),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def paired_metrics(merged: pd.DataFrame, p2_col: str, p3_col: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    participant_rows: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    for repeat in sorted(merged["repeat"].unique()):
        sub = merged.loc[merged["repeat"] == repeat]
        y = sub["y"].to_numpy(dtype=int)
        p2 = sub[p2_col].to_numpy(dtype=np.float64)
        p3 = sub[p3_col].to_numpy(dtype=np.float64)
        loss2 = clipped_log_loss(y, p2)
        loss3 = clipped_log_loss(y, p3)
        delta = loss2 - loss3
        brier2 = (y - p2) ** 2
        brier3 = (y - p3) ** 2
        delta_brier = brier2 - brier3
        auc2 = float(roc_auc_score(y, p2))
        auc3 = float(roc_auc_score(y, p3))
        for pid, yy, pp2, pp3, l2, l3, d, b2, b3, db in zip(
            sub["participant_id"], y, p2, p3, loss2, loss3, delta, brier2, brier3, delta_brier,
        ):
            participant_rows.append({
                "participant_id": pid, "repeat": int(repeat), "y": int(yy),
                "p_m2": float(pp2), "p_m3": float(pp3), "loss_m2": float(l2), "loss_m3": float(l3),
                "delta_ll": float(d), "brier_m2": float(b2), "brier_m3": float(b3), "delta_brier": float(db),
            })
        repeat_rows.append({
            "repeat": int(repeat), "m2_logloss": float(loss2.mean()), "m3_logloss": float(loss3.mean()),
            "delta_ll": float(delta.mean()), "m2_brier": float(brier2.mean()), "m3_brier": float(brier3.mean()),
            "delta_brier": float(delta_brier.mean()), "m2_auroc": auc2, "m3_auroc": auc3,
            "delta_auroc": auc3 - auc2,
        })
    repeat_df = pd.DataFrame(repeat_rows)
    delta = repeat_df["delta_ll"].to_numpy(dtype=np.float64)
    summary = {
        "m2_logloss": float(repeat_df["m2_logloss"].mean()),
        "m3_logloss": float(repeat_df["m3_logloss"].mean()),
        "delta_ll": float(delta.mean()),
        "delta_ll_repeat_summary": metric_summary(delta),
        "delta_brier_repeat_summary": metric_summary(repeat_df["delta_brier"].to_numpy(dtype=np.float64)),
        "delta_auroc_repeat_summary": metric_summary(repeat_df["delta_auroc"].to_numpy(dtype=np.float64)),
        "repeat_count": int(len(repeat_df)),
    }
    return pd.DataFrame(participant_rows), repeat_df, summary


def merge_predictions(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = frames["M2"][[
        "outcome", "repeat", "outer_fold", "participant_index", "participant_id", "y",
        "probability", "representation", "C",
    ]].rename(columns={
        "probability": "p_m2", "representation": "m2_representation", "C": "m2_C",
    })
    adaptive = frames["adaptive_M3"][[
        "repeat", "outer_fold", "participant_index", "probability", "representation", "C",
        "initial_representation", "initial_C", "initial_inner_logloss", "actual_inner_rank",
    ]].rename(columns={
        "probability": "p_adaptive_m3", "representation": "m3_representation", "C": "m3_C",
    })
    merged = base.merge(adaptive, on=["repeat", "outer_fold", "participant_index"], validate="one_to_one")
    for name in ("M0", "M1", "E0", "E1", "E2"):
        extra = frames[name][[
            "repeat", "outer_fold", "participant_index", "probability", "representation", "C",
        ]].rename(columns={
            "probability": f"p_{name.lower()}", "representation": f"{name.lower()}_representation", "C": f"{name.lower()}_C",
        })
        merged = merged.merge(extra, on=["repeat", "outer_fold", "participant_index"], validate="one_to_one")
    return merged.sort_values(["repeat", "participant_index"]).reset_index(drop=True)


def calibration_rows(
    y: np.ndarray, probability: np.ndarray, ids: list[str], repeat: int, model: str,
    failures: list[dict[str, Any]], outcome: str,
) -> list[dict[str, Any]]:
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    z = np.log(p / (1.0 - p))
    rows: list[dict[str, Any]] = []
    base = {"outcome": outcome, "model": model, "repeat": repeat}
    try:
        def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
            eta = params[0] + params[1] * z
            q = special.expit(eta)
            value = float(np.sum(np.logaddexp(0.0, eta) - y * eta))
            gradient = np.asarray([np.sum(q - y), np.sum((q - y) * z)], dtype=np.float64)
            return value, gradient

        fit = optimize.minimize(
            lambda params: objective(params)[0], np.asarray([0.0, 1.0]),
            jac=lambda params: objective(params)[1], method="BFGS",
            options={"gtol": 1e-10, "maxiter": 10000},
        )
        if not fit.success or not np.all(np.isfinite(fit.x)):
            raise RuntimeError(f"CALIBRATION_JOINT_FAILURE:{fit.message}")
        citl = optimize.brentq(
            lambda intercept: np.sum(y - special.expit(intercept + z)),
            -50.0, 50.0, xtol=1e-12, rtol=4.0 * np.finfo(float).eps, maxiter=1000,
        )
        rows.append({
            **base, "record_type": "scalar", "intercept": float(fit.x[0]),
            "slope": float(fit.x[1]), "citl": float(citl), "failure_code": "",
        })
    except Exception as exc:
        failures.append({
            "outcome": outcome, "model_family": model, "repeat": repeat, "outer_fold": "",
            "inner_fold": "", "representation": "", "C": "", "stage": "calibration",
            "error_code": "CALIBRATION_FAILURE", "message": repr(exc),
        })
        rows.append({
            **base, "record_type": "scalar", "intercept": "", "slope": "", "citl": "",
            "failure_code": "CALIBRATION_FAILURE",
        })

    order = sorted(range(len(ids)), key=lambda index: (float(probability[index]), str(ids[index])))
    bin_members: dict[int, list[int]] = {b: [] for b in range(4)}
    for rank, index in enumerate(order):
        bin_members[min(3, int(math.floor(4 * rank / len(ids))))].append(index)
    for bin_id in range(4):
        members = bin_members[bin_id]
        rows.append({
            **base, "record_type": "reliability_bin", "bin": bin_id, "n": len(members),
            "mean_probability": float(np.mean(probability[members])) if members else "",
            "observed_rate": float(np.mean(y[members])) if members else "", "failure_code": "",
        })
    return rows


def summarize_selection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen = [row for row in rows if row.get("model_family") == "adaptive_M3_SELECTED"]
    counts = Counter(str(row["representation"]) for row in chosen)
    return [
        {"representation": rep, "outer_fit_count": int(counts.get(rep, 0)), "proportion": float(counts.get(rep, 0) / 50.0)}
        for rep in REPRESENTATIONS
    ]


def write_secondary_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "analysis", "repeat", "m2_logloss", "representation_logloss", "delta_ll",
        "m2_brier", "representation_brier", "delta_brier",
        "m2_auroc", "representation_auroc", "delta_auroc",
    ]
    write_csv(path, rows, fields)


def run_fixed_secondary(
    merged: pd.DataFrame, fixed: pd.DataFrame, family: str, analysis_name: str,
) -> list[dict[str, Any]]:
    tmp = merged[["repeat", "outer_fold", "participant_index", "y", "p_m2"]].merge(
        fixed[["repeat", "outer_fold", "participant_index", "probability"]].rename(columns={"probability": "p_rep"}),
        on=["repeat", "outer_fold", "participant_index"], validate="one_to_one",
    )
    rows: list[dict[str, Any]] = []
    for repeat in sorted(tmp["repeat"].unique()):
        sub = tmp.loc[tmp["repeat"] == repeat]
        y = sub["y"].to_numpy(dtype=int)
        p2 = sub["p_m2"].to_numpy(dtype=np.float64)
        pr = sub["p_rep"].to_numpy(dtype=np.float64)
        auc2 = float(roc_auc_score(y, p2))
        aucr = float(roc_auc_score(y, pr))
        rows.append({
            "analysis": analysis_name, "repeat": int(repeat),
            "m2_logloss": float(clipped_log_loss(y, p2).mean()),
            "representation_logloss": float(clipped_log_loss(y, pr).mean()),
            "delta_ll": float(np.mean(clipped_log_loss(y, p2) - clipped_log_loss(y, pr))),
            "m2_brier": float(np.mean((y - p2) ** 2)),
            "representation_brier": float(np.mean((y - pr) ** 2)),
            "delta_brier": float(np.mean((y - p2) ** 2 - (y - pr) ** 2)),
            "m2_auroc": auc2, "representation_auroc": aucr, "delta_auroc": aucr - auc2,
        })
    return rows








