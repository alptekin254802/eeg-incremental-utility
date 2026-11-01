"""Spectral and covariance EEG representations using the common preprocessing."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "audit" / "stage1_5_enriched_eeg_feasibility"
STAGE1_DIR = ROOT / "audit" / "stage1_blind_feature_extraction"
STAGE1_SCRIPT = STAGE1_DIR / "run_stage1_extraction.py"
STAGE1_MATRIX = STAGE1_DIR / "BLIND_PRIMARY_FEATURE_MATRIX.csv"
STAGE1_QC = STAGE1_DIR / "EEG_PREPROCESSING_QC.csv"

CHANNELS = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8",
    "FC6", "F4", "F8", "AF4",
]
REGIONS = {
    "frontal": ["AF3", "F7", "F3", "FC5", "FC6", "F4", "F8", "AF4"],
    "temporal": ["T7", "T8"],
    "posterior": ["P7", "O1", "O2", "P8"],
}
BANDS = {
    "delta": (1.0, 3.0),
    "theta": (4.0, 7.0),
    "alpha": (8.0, 12.0),
    "beta": (13.0, 30.0),
}
E2_BANDS = {k: BANDS[k] for k in ("theta", "alpha", "beta")}
QUANTITIES = [
    "log_broadband",
    "log_rel_delta",
    "log_rel_theta",
    "log_rel_alpha",
    "log_rel_beta",
]
E0_QUANTITIES = ["log_broadband", "rel_theta", "rel_alpha", "rel_beta"]
TARGET_FS = 128.0
WINDOW_SAMPLES = 256
WINDOW_STEP = 128
OAS_EIGEN_FLOOR_FACTOR = 1e-6


def load_stage1_module():
    spec = importlib.util.spec_from_file_location("stage1_frozen", STAGE1_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import frozen Stage-1 script: {STAGE1_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()




def csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        return fields, list(reader)


def read_stage1_primary() -> tuple[list[str], list[dict[str, float]], list[str]]:
    fields, rows = csv_rows(STAGE1_MATRIX)
    forbidden = {"group", "diagnosed", "diagnosis", "label", "outcome"}
    if forbidden.intersection({x.strip().lower() for x in fields}):
        raise RuntimeError("Stage-1 primary matrix unexpectedly contains a label/outcome field")
    required = ["participant_id", *[
        f"{region}_{quantity}"
        for region in ("frontal", "temporal", "posterior")
        for quantity in E0_QUANTITIES
    ]]
    missing = [x for x in required if x not in fields]
    if missing:
        raise RuntimeError(f"Stage-1 primary matrix missing frozen E0 columns: {missing}")
    ids: list[str] = []
    values: list[dict[str, float]] = []
    for row in rows:
        pid = row["participant_id"].strip()
        ids.append(pid)
        values.append({x: float(row[x]) for x in required[1:]})
    if len(ids) != 96 or len(set(ids)) != 96:
        raise RuntimeError(f"Expected exactly 96 unique Stage-1 primary IDs; found {len(ids)} rows / {len(set(ids))} unique")
    return ids, values, required[1:]


def read_stage1_qc() -> dict[str, dict[str, str]]:
    fields, rows = csv_rows(STAGE1_QC)
    forbidden = {"group", "diagnosed", "diagnosis", "label", "outcome"}
    if forbidden.intersection({x.strip().lower() for x in fields}):
        raise RuntimeError("Stage-1 EEG QC unexpectedly contains a label/outcome field")
    return {row["participant_id"]: row for row in rows}


def format_float(value: float) -> str:
    return format(float(value), ".17g")


def write_matrix(path: Path, ids: list[str], columns: list[str], matrix: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["participant_id", *columns])
        for pid, row in zip(ids, np.asarray(matrix)):
            writer.writerow([pid, *[format_float(x) for x in row]])


def write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def retain_windows(stage1: Any, segments: list[np.ndarray]) -> tuple[list[np.ndarray], list[tuple[int, int]], dict[str, Any]]:
    """Reproduce Stage-1's window acceptance logic exactly, including union duration."""
    windows: list[np.ndarray] = []
    window_ids: list[tuple[int, int]] = []
    for segment_i, segment in enumerate(segments):
        for start in range(0, segment.shape[0] - WINDOW_SAMPLES + 1, WINDOW_STEP):
            windows.append(segment[start:start + WINDOW_SAMPLES])
            window_ids.append((segment_i, start))
    if not windows:
        return [], [], {
            "candidate_window_count": 0, "rejected_window_count": 0,
            "retained_window_count": 0, "clean_window_fraction": 0.0,
            "clean_seconds": 0.0, "artifact_threshold_robust": np.nan,
        }
    ptp = np.asarray([
        np.ptp(signal.detrend(window, axis=0, type="linear"), axis=0)
        for window in windows
    ])
    robust_limit = stage1.robust_threshold(ptp.ravel(), stage1.ROBUST_MAD_Z)
    rejected = (ptp > stage1.ABSOLUTE_PTP_UV).any(axis=1) | (ptp > robust_limit).any(axis=1)
    retained = [windows[i] for i in range(len(windows)) if not rejected[i]]
    retained_ids = [window_ids[i] for i in range(len(windows)) if not rejected[i]]
    intervals: dict[int, list[tuple[int, int]]] = {}
    for seg_i, start in retained_ids:
        intervals.setdefault(seg_i, []).append((start, start + WINDOW_SAMPLES))
    clean_samples = 0
    for pairs in intervals.values():
        pairs = sorted(pairs)
        cur_start, cur_stop = pairs[0]
        for start, stop in pairs[1:]:
            if start <= cur_stop:
                cur_stop = max(cur_stop, stop)
            else:
                clean_samples += cur_stop - cur_start
                cur_start, cur_stop = start, stop
        clean_samples += cur_stop - cur_start
    qc = {
        "candidate_window_count": int(len(windows)),
        "rejected_window_count": int(rejected.sum()),
        "retained_window_count": int(len(retained)),
        "clean_window_fraction": float(len(retained) / len(windows)),
        "clean_seconds": float(clean_samples / TARGET_FS),
        "artifact_threshold_robust": float(robust_limit),
        "artifact_absolute_threshold_uv": float(stage1.ABSOLUTE_PTP_UV),
    }
    return retained, retained_ids, qc


def integrate_psd(frequencies: np.ndarray, values: np.ndarray, low: float, high: float) -> np.ndarray:
    mask = (frequencies >= low) & (frequencies <= high)
    if int(mask.sum()) < 2:
        return np.full(values.shape[:-1], np.nan)
    return np.trapezoid(values[..., mask, :], frequencies[mask], axis=-2)


def spectral_arrays(windows: list[np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    stacked = np.stack(windows, axis=0)
    frequencies, psd = signal.welch(
        stacked, fs=TARGET_FS, window="hann", nperseg=WINDOW_SAMPLES,
        noverlap=0, detrend="constant", scaling="density", average="mean", axis=1,
    )
    broadband = integrate_psd(frequencies, psd, 1.0, 40.0)
    band_values = {
        band: integrate_psd(frequencies, psd, low, high)
        for band, (low, high) in BANDS.items()
    }
    quantities = {
        "log_broadband": np.log10(broadband),
        "log_rel_delta": np.log10(band_values["delta"] / broadband),
        "log_rel_theta": np.log10(band_values["theta"] / broadband),
        "log_rel_alpha": np.log10(band_values["alpha"] / broadband),
        "log_rel_beta": np.log10(band_values["beta"] / broadband),
    }
    return frequencies, quantities, psd


def e0_from_spectral(stage1: Any, windows: list[np.ndarray]) -> dict[str, float]:
    """Compute the frozen E0 formulas independently for exact agreement checks."""
    frequencies, psd_quantities, _ = spectral_arrays(windows)
    broadband = integrate_psd(
        frequencies,
        signal.welch(
            np.stack(windows, axis=0), fs=TARGET_FS, window="hann",
            nperseg=WINDOW_SAMPLES, noverlap=0, detrend="constant",
            scaling="density", average="mean", axis=1,
        )[1],
        1.0, 40.0,
    )
    # The following band integrations are deliberately kept in the same
    # array/order convention as the frozen Stage-1 implementation.
    _, psd = signal.welch(
        np.stack(windows, axis=0), fs=TARGET_FS, window="hann", nperseg=WINDOW_SAMPLES,
        noverlap=0, detrend="constant", scaling="density", average="mean", axis=1,
    )
    raw = {
        "log_broadband": np.log10(np.nanmedian(broadband, axis=0)),
        "rel_theta": np.nanmedian(integrate_psd(frequencies, psd, 4.0, 7.0) / broadband, axis=0),
        "rel_alpha": np.nanmedian(integrate_psd(frequencies, psd, 8.0, 12.0) / broadband, axis=0),
        "rel_beta": np.nanmedian(integrate_psd(frequencies, psd, 13.0, 30.0) / broadband, axis=0),
    }
    channel_index = {name: i for i, name in enumerate(CHANNELS)}
    out: dict[str, float] = {}
    for region, names in REGIONS.items():
        indices = [channel_index[name] for name in names]
        for quantity in E0_QUANTITIES:
            out[f"{region}_{quantity}"] = float(np.nanmedian(raw[quantity][indices]))
    # Avoid an unused local warning while making the implementation intent
    # explicit: psd_quantities is computed by the exact E1 branch below.
    _ = psd_quantities
    return out


def e1_features(windows: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    frequencies, quantities, psd = spectral_arrays(windows)
    topo_columns = [f"{channel}_{quantity}" for channel in CHANNELS for quantity in QUANTITIES]
    topo = np.asarray([
        [np.median(quantities[quantity][:, channel_i]) for channel_i in range(len(CHANNELS)) for quantity in QUANTITIES]
    ])
    topo = topo.reshape(1, -1)

    channel_index = {name: i for i, name in enumerate(CHANNELS)}
    temporal_columns: list[str] = []
    temporal_values: list[float] = []
    for region, names in REGIONS.items():
        indices = [channel_index[name] for name in names]
        for quantity in QUANTITIES:
            regional = np.median(quantities[quantity][:, indices], axis=1)
            q25, q75 = np.quantile(regional, [0.25, 0.75], method="linear")
            temporal_columns.append(f"{region}_iqr_{quantity}")
            temporal_values.append(float(q75 - q25))
    temporal = np.asarray(temporal_values, dtype=float).reshape(1, -1)

    entropy_columns: list[str] = []
    entropy_values: list[float] = []
    freq_mask = (frequencies >= 1.0) & (frequencies <= 40.0)
    channel_index = {name: i for i, name in enumerate(CHANNELS)}
    for region, names in REGIONS.items():
        indices = [channel_index[name] for name in names]
        regional_psd = np.median(psd[:, freq_mask, :][:, :, indices], axis=(0, 2))
        probabilities = regional_psd / np.sum(regional_psd)
        entropy = -np.sum(probabilities * np.log(probabilities)) / np.log(float(probabilities.size))
        entropy_columns.append(f"{region}_spectral_entropy")
        entropy_values.append(float(entropy))
    entropy = np.asarray(entropy_values, dtype=float).reshape(1, -1)
    return topo, temporal, entropy, topo_columns, temporal_columns, entropy_columns


def helmert_contrast_basis(n: int) -> np.ndarray:
    q = np.zeros((n, n - 1), dtype=float)
    for k in range(1, n):
        denominator = math.sqrt(k * (k + 1))
        q[:k, k - 1] = 1.0 / denominator
        q[k, k - 1] = -k / denominator
    if not np.allclose(q.T @ q, np.eye(n - 1), rtol=0.0, atol=1e-14):
        raise RuntimeError("CAR contrast basis failed orthonormality check")
    if not np.allclose(q.T @ np.ones(n), np.zeros(n - 1), rtol=0.0, atol=1e-14):
        raise RuntimeError("CAR contrast basis failed zero-sum check")
    return q


def oas_covariance(x: np.ndarray) -> tuple[np.ndarray, float]:
    n_samples, n_features = x.shape
    sample = np.cov(x, rowvar=False, ddof=1)
    sample = (sample + sample.T) / 2.0
    mu = float(np.trace(sample) / n_features)
    alpha = float(np.mean(sample * sample))
    numerator = alpha + mu * mu
    denominator = (n_samples + 1.0) * (alpha - (mu * mu / n_features))
    if denominator <= 0.0 or not np.isfinite(denominator):
        shrinkage = 1.0
    else:
        shrinkage = float(numerator / denominator)
    shrinkage = float(np.clip(shrinkage, 0.0, 1.0))
    shrunk = (1.0 - shrinkage) * sample + shrinkage * mu * np.eye(n_features)
    shrunk = (shrunk + shrunk.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(shrunk)
    floor = OAS_EIGEN_FLOOR_FACTOR * float(np.trace(shrunk)) / n_features
    eigenvalues = np.maximum(eigenvalues, floor)
    shrunk = (eigenvectors * eigenvalues) @ eigenvectors.T
    shrunk = (shrunk + shrunk.T) / 2.0
    shrunk /= float(np.trace(shrunk))
    return shrunk, shrinkage


def spd_log(covariance: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh((covariance + covariance.T) / 2.0)
    if np.any(values <= 0.0) or not np.all(np.isfinite(values)):
        raise RuntimeError("Non-SPD covariance passed to matrix logarithm")
    return (vectors * np.log(values)) @ vectors.T


def spd_exp(symmetric: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh((symmetric + symmetric.T) / 2.0)
    result = (vectors * np.exp(values)) @ vectors.T
    result = (result + result.T) / 2.0
    result /= float(np.trace(result))
    return result


def e2_covariance(
    segments: list[np.ndarray], retained_ids: list[tuple[int, int]], q_car: np.ndarray,
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    covariances: list[np.ndarray] = []
    qcs: dict[str, dict[str, float]] = {}
    band_filters = {
        band: signal.butter(4, limits, btype="bandpass", fs=TARGET_FS, output="sos")
        for band, limits in E2_BANDS.items()
    }
    filtered_segments = {
        band: [signal.sosfiltfilt(sos, segment, axis=0) for segment in segments]
        for band, sos in band_filters.items()
    }
    for band in E2_BANDS:
        window_covariances: list[np.ndarray] = []
        shrinkages: list[float] = []
        for seg_i, start in retained_ids:
            window = filtered_segments[band][seg_i][start:start + WINDOW_SAMPLES]
            window = window - np.mean(window, axis=0, keepdims=True)
            projected = window @ q_car
            cov, shrinkage = oas_covariance(projected)
            if not np.all(np.isfinite(cov)):
                raise RuntimeError(f"Nonfinite window covariance for {band}")
            window_covariances.append(cov)
            shrinkages.append(shrinkage)
        log_mean = np.mean(np.stack([spd_log(c) for c in window_covariances], axis=0), axis=0)
        participant_cov = spd_exp(log_mean)
        eig = np.linalg.eigvalsh(participant_cov)
        qcs[band] = {
            "window_count": float(len(window_covariances)),
            "oas_shrinkage_min": float(min(shrinkages)),
            "oas_shrinkage_max": float(max(shrinkages)),
            "trace": float(np.trace(participant_cov)),
            "min_eigenvalue": float(eig.min()),
            "max_eigenvalue": float(eig.max()),
            "symmetric_max_abs_error": float(np.max(np.abs(participant_cov - participant_cov.T))),
            "finite": 1.0 if np.all(np.isfinite(participant_cov)) else 0.0,
            "spd": 1.0 if bool(np.all(eig > 0.0)) else 0.0,
        }
        covariances.append(participant_cov)
    return np.stack(covariances, axis=0), qcs


def extract_once(stage1: Any, records_by_id: dict[str, dict[str, Any]], ids: list[str], stage1_values: list[dict[str, float]], stage1_qc: dict[str, dict[str, str]], q_car: np.ndarray) -> dict[str, Any]:
    e0_matrix: list[list[float]] = []
    e0_stage1_matrix: list[list[float]] = []
    topo_rows: list[np.ndarray] = []
    temporal_rows: list[np.ndarray] = []
    entropy_rows: list[np.ndarray] = []
    covariance_rows: list[np.ndarray] = []
    covariance_qc_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    e0_keys = [f"{region}_{quantity}" for region in ("frontal", "temporal", "posterior") for quantity in E0_QUANTITIES]
    for pid, frozen_values in zip(ids, stage1_values):
        record = records_by_id[pid]
        eeg_path = ROOT / Path(record["eeg_path"])
        metadata, data, timestamps, _counters, interp, _gaps = stage1.parse_eeg(eeg_path)
        bad_indices, _bad_detail = stage1.bad_channels_from_raw(data, float(metadata["sampling_rate_hz"]))
        segments, _bounds, _preprocess_detail = stage1.resample_and_preprocess(
            data, timestamps, interp, bad_indices, float(metadata["sampling_rate_hz"]),
        )
        windows, retained_ids, window_qc = retain_windows(stage1, segments)
        frozen_qc = stage1_qc[pid]
        for key in ("candidate_window_count", "rejected_window_count"):
            if int(frozen_qc[key]) != int(window_qc[key]):
                raise RuntimeError(f"Window-accounting mismatch for {pid}: {key}")
        for key in ("clean_window_fraction", "clean_seconds"):
            if not math.isclose(float(frozen_qc[key]), float(window_qc[key]), rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(f"Window-accounting mismatch for {pid}: {key}")
        if not windows:
            raise RuntimeError(f"No accepted Stage-1 windows for fixed eligible participant {pid}")

        frozen_e0, _frozen_artifact_qc = stage1.artifact_and_features(segments)
        local_e0 = e0_from_spectral(stage1, windows)
        e0_matrix.append([local_e0[k] for k in e0_keys])
        e0_stage1_matrix.append([frozen_e0[k] for k in e0_keys])
        topo, temporal, entropy, topo_columns, temporal_columns, entropy_columns = e1_features(windows)
        topo_rows.append(topo[0])
        temporal_rows.append(temporal[0])
        entropy_rows.append(entropy[0])
        covariances, covariance_qc = e2_covariance(segments, retained_ids, q_car)
        covariance_rows.append(covariances)
        for band in E2_BANDS:
            covariance_qc_rows.append({"participant_id": pid, "band": band, **covariance_qc[band]})
        qc_rows.append({
            "participant_id": pid,
            "session_id": record["session_id"],
            "eeg_source_file": record["eeg_source_file"],
            "sampling_rate_hz": metadata["sampling_rate_hz"],
            "device_model": metadata.get("device_model", ""),
            "channel_count": metadata.get("channel_count", ""),
            "candidate_window_count": window_qc["candidate_window_count"],
            "rejected_window_count": window_qc["rejected_window_count"],
            "retained_window_count": window_qc["retained_window_count"],
            "clean_window_fraction": window_qc["clean_window_fraction"],
            "clean_seconds": window_qc["clean_seconds"],
            "e0_finite": bool(np.all(np.isfinite(e0_matrix[-1]))),
            "e1_topographic_finite": bool(np.all(np.isfinite(topo[0]))),
            "e1_temporal_iqr_finite": bool(np.all(np.isfinite(temporal[0]))),
            "e1_entropy_finite": bool(np.all(np.isfinite(entropy[0]))),
            "theta_min_eigenvalue": covariance_qc["theta"]["min_eigenvalue"],
            "alpha_min_eigenvalue": covariance_qc["alpha"]["min_eigenvalue"],
            "beta_min_eigenvalue": covariance_qc["beta"]["min_eigenvalue"],
        })
    return {
        "ids": ids,
        "e0": np.asarray(e0_matrix),
        "e0_stage1": np.asarray(e0_stage1_matrix),
        "topo": np.asarray(topo_rows),
        "temporal": np.asarray(temporal_rows),
        "entropy": np.asarray(entropy_rows),
        "covariances": np.asarray(covariance_rows),
        "covariance_qc_rows": covariance_qc_rows,
        "qc_rows": qc_rows,
        "topo_columns": topo_columns,
        "temporal_columns": temporal_columns,
        "entropy_columns": entropy_columns,
        "e0_columns": e0_keys,
    }


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def rank_info(matrix: np.ndarray, candidate: int) -> dict[str, Any]:
    x = np.asarray(matrix, dtype=float)
    centered = x - np.mean(x, axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    tolerance = float(max(centered.shape) * np.finfo(float).eps * (float(singular_values[0]) if singular_values.size else 0.0))
    rank = int(np.sum(singular_values > tolerance)) if singular_values.size else 0
    return {
        "finite": bool(np.all(np.isfinite(x))),
        "rows": int(x.shape[0]), "columns": int(x.shape[1]),
        "centered_rank": rank, "candidate_components": candidate,
        "rank_sufficient": bool(rank >= candidate),
        "svd_tolerance": tolerance,
        "minimum_singular_value": float(singular_values[-1]) if singular_values.size else 0.0,
    }










