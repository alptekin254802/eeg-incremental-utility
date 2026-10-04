"""EEG feature extraction and participant-level evaluation of post-hoc representations.

Feature extraction uses signal measurements without study-group labels. Model
evaluation uses the primary participant roster and nested cross-validation folds."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import platform
import shutil
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal.windows import dpss
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
STAGE1_DIR = ROOT / "audit" / "stage1_blind_feature_extraction"
STAGE1_CODE = STAGE1_DIR / "run_stage1_extraction.py"
STAGE1_MATRIX = STAGE1_DIR / "BLIND_PRIMARY_FEATURE_MATRIX.csv"
STAGE1_QC = STAGE1_DIR / "EEG_PREPROCESSING_QC.csv"
PREREG_DIR = ROOT / "audit" / "preregistered_analysis"

CHANNELS = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8",
    "FC6", "F4", "F8", "AF4",
]
REGIONS = {
    "frontal": ["AF3", "F7", "F3", "FC5", "FC6", "F4", "F8", "AF4"],
    "temporal": ["T7", "T8"],
    "posterior": ["P7", "O1", "O2", "P8"],
}
BANDS = {"theta": (4.0, 7.0), "alpha": (8.0, 12.0), "beta": (13.0, 30.0)}
BAND_ORDER = ["theta", "alpha", "beta"]
FS = 128.0
WINDOW_SAMPLES = 256
WINDOW_STEP = 128
E3_TAPERS = dpss(256, NW=2.5, Kmax=4, sym=False, norm=2)
PAIR_LIST = [(i, j) for i in range(13) for j in range(i + 1, 14)]
PAIR_NAMES = [(CHANNELS[i], CHANNELS[j]) for i, j in PAIR_LIST]
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
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


class FeasibilityFailure(RuntimeError):
    pass


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load frozen module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            clean = {}
            for field in fields:
                value = row.get(field, "")
                if isinstance(value, np.generic):
                    value = value.item()
                if isinstance(value, float) and not math.isfinite(value):
                    value = ""
                clean[field] = value
            writer.writerow(clean)




def stage1_capture(stage1, record: dict[str, Any], demographics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Execute the frozen Stage-1 path and retain accepted windows in memory."""
    participant = record["participant_id"]
    if participant not in demographics or demographics[participant]["age"] is None:
        raise RuntimeError(f"ALLOWED_DEMOGRAPHICS_MISSING|{participant}")
    metadata, data, timestamps, _counters, interp, _gaps = stage1.parse_eeg(record["eeg_path"])
    bad_indices, bad_details = stage1.bad_channels_from_raw(data, float(metadata["sampling_rate_hz"]))
    bad_names = [stage1.EEG_CHANNELS[i] for i in bad_indices]
    qc: dict[str, Any] = {
        "participant_id": participant,
        "session_id": record["session_id"],
        "native_sampling_rate_hz": metadata["sampling_rate_hz"],
        "device": metadata["device_model"],
        "bad_channel_count": len(bad_indices),
        "bad_channel_names": ";".join(bad_names),
        "preprocessing_status": "",
        "exclusion_reason": "",
    }
    if metadata["structural_parser_status"] != "PASS" or metadata["channel_count"] != 14 or metadata["sampling_rate_hz"] not in {128.0, 256.0}:
        qc.update({"preprocessing_status": "FAIL_STRUCTURAL", "exclusion_reason": "parser, channel, timestamp, or sampling metadata failure"})
        return {"record": record, "qc": qc, "segments": [], "accepted_windows": [], "accepted_ids": [], "e0": {}}
    if len(bad_indices) > stage1.MAX_BAD_CHANNELS:
        qc.update({"preprocessing_status": "FAIL_TOO_MANY_BAD_CHANNELS", "exclusion_reason": ">2 frozen bad channels"})
        return {"record": record, "qc": qc, "segments": [], "accepted_windows": [], "accepted_ids": [], "e0": {}}

    segments, processed_bounds, prep_detail = stage1.resample_and_preprocess(
        data, timestamps, interp, bad_indices, float(metadata["sampling_rate_hz"])
    )
    e0, artifact_qc = stage1.artifact_and_features(segments)
    accepted_windows, accepted_ids, local_qc = accepted_window_payload(stage1, segments)
    for key in ("candidate_window_count", "rejected_window_count", "retained_window_count", "clean_window_fraction", "clean_seconds"):
        if not math.isclose(float(artifact_qc.get(key, np.nan)), float(local_qc.get(key, np.nan)), rel_tol=0.0, abs_tol=0.0):
            raise RuntimeError(f"STAGE1_ACCEPTED_WINDOW_RECONSTRUCTION_MISMATCH|{participant}|{key}")
    qc.update(artifact_qc)
    qc.update({
        "segment_count": len(segments),
        "processed_bounds": processed_bounds,
        "final_usable_channels": max(0, len(stage1.EEG_CHANNELS) - len(bad_indices)),
        "frontal_usable_channels": sum(c not in bad_names for c in stage1.REGIONS["frontal"]),
        "temporal_usable_channels": sum(c not in bad_names for c in stage1.REGIONS["temporal"]),
        "posterior_usable_channels": sum(c not in bad_names for c in stage1.REGIONS["posterior"]),
    })
    if not segments or qc["candidate_window_count"] == 0:
        qc.update({"preprocessing_status": "FAIL_CLEAN_DURATION", "exclusion_reason": "no usable clean 2-second windows"})
    elif qc["clean_seconds"] < stage1.MIN_CLEAN_SECONDS:
        qc.update({"preprocessing_status": "FAIL_CLEAN_DURATION", "exclusion_reason": f"clean_seconds<{stage1.MIN_CLEAN_SECONDS}"})
    elif qc["clean_window_fraction"] < stage1.MIN_CLEAN_WINDOW_FRACTION:
        qc.update({"preprocessing_status": "FAIL_CLEAN_WINDOW_FRACTION", "exclusion_reason": f"clean_window_fraction<{stage1.MIN_CLEAN_WINDOW_FRACTION}"})
    elif qc["final_usable_channels"] < stage1.MIN_USABLE_CHANNELS:
        qc.update({"preprocessing_status": "FAIL_REGION_COVERAGE", "exclusion_reason": f"final_usable_channels<{stage1.MIN_USABLE_CHANNELS}"})
    elif any(qc[f"{region}_usable_channels"] < 1 for region in ("frontal", "temporal", "posterior")):
        qc.update({"preprocessing_status": "FAIL_REGION_COVERAGE", "exclusion_reason": "missing usable frozen region"})
    elif not e0 or any(not math.isfinite(float(value)) for value in e0.values()):
        qc.update({"preprocessing_status": "OTHER_FROZEN_RULE_FAILURE", "exclusion_reason": "undefined primary EEG feature"})
    else:
        qc.update({"preprocessing_status": "PASS", "exclusion_reason": ""})
    qc["preprocessing_detail"] = {**prep_detail, **bad_details}
    if qc["preprocessing_status"] != "PASS":
        accepted_windows, accepted_ids = [], []
    return {
        "record": record, "qc": qc, "segments": segments, "accepted_windows": accepted_windows,
        "accepted_ids": accepted_ids, "e0": e0,
    }


def accepted_window_payload(stage1, segments: list[np.ndarray]) -> tuple[list[np.ndarray], list[tuple[int, int]], dict[str, Any]]:
    windows: list[np.ndarray] = []
    window_ids: list[tuple[int, int]] = []
    for segment_i, segment in enumerate(segments):
        for start in range(0, segment.shape[0] - stage1.WINDOW_SAMPLES + 1, stage1.WINDOW_STEP):
            windows.append(segment[start:start + stage1.WINDOW_SAMPLES])
            window_ids.append((segment_i, start))
    if not windows:
        return [], [], {"candidate_window_count": 0, "rejected_window_count": 0, "retained_window_count": 0, "clean_window_fraction": 0.0, "clean_seconds": 0.0}
    ptp = np.asarray([np.ptp(signal.detrend(window, axis=0, type="linear"), axis=0) for window in windows])
    robust_limit = stage1.robust_threshold(ptp.ravel(), stage1.ROBUST_MAD_Z)
    rejected = (ptp > stage1.ABSOLUTE_PTP_UV).any(axis=1) | (ptp > robust_limit).any(axis=1)
    retained = [windows[i] for i in range(len(windows)) if not rejected[i]]
    retained_ids = [window_ids[i] for i in range(len(windows)) if not rejected[i]]
    intervals: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for segment_i, start in retained_ids:
        intervals[segment_i].append((start, start + stage1.WINDOW_SAMPLES))
    clean_samples = 0
    for segment_i, pairs in intervals.items():
        pairs = sorted(pairs)
        cur_start, cur_stop = pairs[0]
        for start, stop in pairs[1:]:
            if start <= cur_stop:
                cur_stop = max(cur_stop, stop)
            else:
                clean_samples += cur_stop - cur_start
                cur_start, cur_stop = start, stop
        clean_samples += cur_stop - cur_start
    return retained, retained_ids, {
        "candidate_window_count": len(windows),
        "rejected_window_count": int(rejected.sum()),
        "retained_window_count": len(retained),
        "clean_window_fraction": float(len(retained) / len(windows)),
        "clean_seconds": float(clean_samples / FS),
    }


def e3_extract(windows: list[np.ndarray]) -> dict[str, float]:
    if len(windows) < 2:
        raise FeasibilityFailure("E3_LESS_THAN_TWO_ACCEPTED_WINDOWS")
    data = np.stack(windows, axis=0).astype(np.float64)
    if not np.all(np.isfinite(data)):
        raise FeasibilityFailure("E3_NONFINITE_WINDOW_DATA")
    data = data - data.mean(axis=1, keepdims=True)
    tapered = data[:, None, :, :] * E3_TAPERS[None, :, :, None]
    spectra = np.fft.rfft(tapered, n=256, axis=2)
    freqs = np.fft.rfftfreq(256, d=1.0 / FS)
    eps = np.finfo(float).eps
    output: dict[str, float] = {}
    for band, (low, high) in BANDS.items():
        bins = np.flatnonzero((freqs >= low) & (freqs <= high))
        for pair_index, (i, j) in enumerate(PAIR_LIST):
            values = []
            cross = spectra[:, :, :, i] * np.conj(spectra[:, :, :, j])
            x = np.imag(cross)
            for bin_index in bins:
                values_x = x[:, :, int(bin_index)].reshape(-1)
                sum_x = float(values_x.sum())
                sum_x2 = float(np.square(values_x).sum())
                sum_abs = float(np.abs(values_x).sum())
                denominator = (sum_abs * sum_abs) - sum_x2
                threshold = eps * max(1.0, sum_abs * sum_abs)
                if not math.isfinite(denominator) or denominator <= threshold:
                    raise FeasibilityFailure(f"E3_UNDEFINED_DWPLI2|{band}|pair={pair_index}|bin={int(bin_index)}")
                numerator = (sum_x * sum_x) - sum_x2
                value = numerator / denominator
                if not math.isfinite(value):
                    raise FeasibilityFailure(f"E3_NONFINITE_DWPLI2|{band}|pair={pair_index}|bin={int(bin_index)}")
                values.append(value)
            name = f"{band}_dwpli2_{CHANNELS[i]}_{CHANNELS[j]}"
            output[name] = float(np.median(np.asarray(values, dtype=np.float64)))
    if len(output) != 273 or not np.all(np.isfinite(np.asarray(list(output.values()), dtype=float))):
        raise FeasibilityFailure("E3_OUTPUT_DIMENSION_OR_FINITE_FAILURE")
    return output


def permutation_entropy(values: np.ndarray) -> float:
    vectors = np.lib.stride_tricks.sliding_window_view(values, 5)
    orders = np.argsort(vectors, axis=1, kind="stable")
    codes = np.ravel_multi_index(orders.T, (5, 5, 5, 5, 5))
    counts = np.bincount(codes, minlength=3125).astype(np.float64)
    probabilities = counts[counts > 0] / float(len(codes))
    return float(-np.sum(probabilities * np.log(probabilities)) / np.log(120.0))


def e4_extract(windows: list[np.ndarray]) -> dict[str, float]:
    if len(windows) < 2:
        raise FeasibilityFailure("E4_LESS_THAN_TWO_ACCEPTED_WINDOWS")
    data = np.stack(windows, axis=0).astype(np.float64)
    if not np.all(np.isfinite(data)):
        raise FeasibilityFailure("E4_NONFINITE_WINDOW_DATA")
    per_window = np.full((data.shape[0], data.shape[2], 3), np.nan, dtype=float)
    for window_index, window in enumerate(data):
        med = np.median(window, axis=0)
        mad = np.median(np.abs(window - med[None, :]), axis=0)
        scale = 1.4826 * mad
        if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
            raise FeasibilityFailure(f"E4_ZERO_OR_NONFINITE_ROBUST_SCALE|window={window_index}")
        normalized = (window - med[None, :]) / scale[None, :]
        for channel_index in range(data.shape[2]):
            x = normalized[:, channel_index]
            variance = float(np.var(x, ddof=0))
            d1 = np.diff(x)
            d2 = np.diff(x, n=2)
            variance_d1 = float(np.var(d1, ddof=0))
            variance_d2 = float(np.var(d2, ddof=0))
            if not all(math.isfinite(v) and v > 0 for v in (variance, variance_d1, variance_d2)):
                raise FeasibilityFailure(f"E4_HJORTH_UNDEFINED|window={window_index}|channel={channel_index}")
            mobility = math.sqrt(variance_d1 / variance)
            complexity = math.sqrt(variance_d2 / variance_d1) / mobility
            entropy = permutation_entropy(x)
            if not all(math.isfinite(v) for v in (entropy, mobility, complexity)):
                raise FeasibilityFailure(f"E4_NONFINITE_STATISTIC|window={window_index}|channel={channel_index}")
            per_window[window_index, channel_index, :] = (entropy, mobility, complexity)
    channel_medians = np.median(per_window, axis=0)
    output: dict[str, float] = {}
    channel_index = {name: i for i, name in enumerate(CHANNELS)}
    for region, region_channels in REGIONS.items():
        indices = [channel_index[name] for name in region_channels]
        for quantity_index, quantity in enumerate(("perm_entropy", "hjorth_mobility", "hjorth_complexity")):
            output[f"{region}_{quantity}"] = float(np.median(channel_medians[indices, quantity_index]))
    if len(output) != 9 or not np.all(np.isfinite(np.asarray(list(output.values()), dtype=float))):
        raise FeasibilityFailure("E4_OUTPUT_DIMENSION_OR_FINITE_FAILURE")
    return output


def e5_extract(segments: list[np.ndarray], accepted_ids: list[tuple[int, int]]) -> dict[str, float]:
    if not accepted_ids:
        raise FeasibilityFailure("E5_NO_ACCEPTED_WINDOWS")
    accepted_by_segment: dict[int, set[int]] = defaultdict(set)
    for segment_index, start in accepted_ids:
        accepted_by_segment[segment_index].add(start)
    trajectories: dict[tuple[str, str], list[tuple[int, int, float]]] = defaultdict(list)
    channel_index = {name: i for i, name in enumerate(CHANNELS)}
    region_indices = {region: [channel_index[name] for name in names] for region, names in REGIONS.items()}
    for segment_index, segment in enumerate(segments):
        valid_starts = []
        for frame_start in range(0, segment.shape[0] - 128 + 1, 32):
            if any(window_start <= frame_start and frame_start + 128 <= window_start + 256 for window_start in accepted_by_segment.get(segment_index, set())):
                valid_starts.append(frame_start)
        for frame_start in valid_starts:
            frame = segment[frame_start:frame_start + 128]
            frequencies, psd = signal.periodogram(
                frame, fs=FS, window="hann", nfft=128, detrend="constant",
                scaling="density", return_onesided=True, axis=0,
            )
            band_energy = {}
            for band, (low, high) in BANDS.items():
                broad_mask = (frequencies >= 1.0) & (frequencies <= 40.0)
                band_mask = (frequencies >= low) & (frequencies <= high)
                broad = np.trapezoid(psd[broad_mask, :], frequencies[broad_mask], axis=0)
                power = np.trapezoid(psd[band_mask, :], frequencies[band_mask], axis=0)
                if np.any(~np.isfinite(broad)) or np.any(~np.isfinite(power)) or np.any(broad <= 0) or np.any(power <= 0):
                    raise FeasibilityFailure(f"E5_NONPOSITIVE_OR_NONFINITE_POWER|segment={segment_index}|frame={frame_start}|band={band}")
                band_energy[band] = np.log10(power / broad)
            for region, indices in region_indices.items():
                for band in BAND_ORDER:
                    value = float(np.median(band_energy[band][indices]))
                    if not math.isfinite(value):
                        raise FeasibilityFailure(f"E5_NONFINITE_REGION_TRAJECTORY|segment={segment_index}|frame={frame_start}")
                    trajectories[(region, band)].append((segment_index, frame_start, value))
    output: dict[str, float] = {}
    for region in REGIONS:
        for band in BAND_ORDER:
            values = trajectories[(region, band)]
            if len(values) < 2:
                raise FeasibilityFailure(f"E5_TOO_FEW_VALID_FRAMES|{region}|{band}")
            arr = np.asarray([item[2] for item in values], dtype=float)
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med)))
            scale = 1.4826 * mad
            if not math.isfinite(scale) or scale <= 0:
                raise FeasibilityFailure(f"E5_ZERO_OR_NONFINITE_TRAJECTORY_MAD|{region}|{band}")
            states = arr.copy()
            high = (states - med) / scale > 1.0
            occupancy = float(np.mean(high))
            runs: list[int] = []
            current = 0
            adjacent_pairs = 0
            changes = 0
            previous_segment = None
            previous_start = None
            previous_high = None
            for (segment_index, frame_start, _), is_high in zip(values, high):
                adjacent = previous_segment == segment_index and previous_start is not None and frame_start - previous_start == 32
                if adjacent:
                    adjacent_pairs += 1
                    if bool(is_high) != bool(previous_high):
                        changes += 1
                if is_high:
                    if current and adjacent and previous_high:
                        current += 1
                    else:
                        if current:
                            runs.append(current)
                        current = 1
                else:
                    if current:
                        runs.append(current)
                        current = 0
                previous_segment = segment_index
                previous_start = frame_start
                previous_high = bool(is_high)
            if current:
                runs.append(current)
            median_duration = float(np.median([1.0 + 0.25 * (run - 1) for run in runs])) if runs else 0.0
            switching = float(changes / (0.25 * adjacent_pairs) * 60.0) if adjacent_pairs else 0.0
            output[f"{region}_{band}_occupancy"] = occupancy
            output[f"{region}_{band}_median_high_state_duration_seconds"] = median_duration
            output[f"{region}_{band}_switching_rate_per_minute"] = switching
    if len(output) != 27 or not np.all(np.isfinite(np.asarray(list(output.values()), dtype=float))):
        raise FeasibilityFailure("E5_OUTPUT_DIMENSION_OR_FINITE_FAILURE")
    return output


def feature_columns(prefix: str, first: dict[str, float]) -> list[str]:
    return list(first.keys())


def centered_rank(matrix: np.ndarray) -> tuple[int, float, float]:
    values = np.asarray(matrix, dtype=np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    if singular.size == 0 or singular[0] == 0:
        return 0, 0.0, 0.0
    tolerance = max(values.shape) * np.finfo(float).eps * singular[0]
    return int(np.sum(singular > tolerance)), float(tolerance), float(singular[0])




def clipped_log_loss(y: np.ndarray, probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), 1e-8, 1.0 - 1e-8)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def transform_block(train: np.ndarray, assessment: np.ndarray, components: int | None, name: str) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train, dtype=np.float64).copy()
    assessment = np.asarray(assessment, dtype=np.float64).copy()
    missing_rate = np.mean(~np.isfinite(train), axis=0)
    if np.any(missing_rate > 0.10):
        raise FeasibilityFailure(f"{name}_MISSINGNESS_GT_10_PERCENT")
    if np.any(~np.isfinite(train)) or np.any(~np.isfinite(assessment)):
        medians = np.nanmedian(train, axis=0)
        if np.any(~np.isfinite(medians)):
            raise FeasibilityFailure(f"{name}_NONFINITE_MEDIAN")
        for column in range(train.shape[1]):
            train[~np.isfinite(train[:, column]), column] = medians[column]
            assessment[~np.isfinite(assessment[:, column]), column] = medians[column]
    scaler = StandardScaler(with_mean=True, with_std=True)
    train_scaled = scaler.fit_transform(train)
    assessment_scaled = scaler.transform(assessment)
    if components is None:
        return train_scaled.astype(float), assessment_scaled.astype(float)
    pca = PCA(n_components=components, svd_solver="full", whiten=False)
    train_pc = pca.fit_transform(train_scaled)
    assessment_pc = pca.transform(assessment_scaled)
    singular = pca.singular_values_
    tolerance = max(train_scaled.shape) * np.finfo(float).eps * singular[0] if len(singular) else np.inf
    if int(np.sum(singular > tolerance)) < components:
        raise FeasibilityFailure(f"{name}_RANK_BELOW_{components}")
    return train_pc.astype(float), assessment_pc.astype(float)


def model_features(X_m2: np.ndarray, X_raw: np.ndarray, train_idx: np.ndarray, assessment_idx: np.ndarray, family: str) -> tuple[np.ndarray, np.ndarray]:
    if family == "E4":
        eeg_train, eeg_assess = X_raw[train_idx], X_raw[assessment_idx]
    else:
        width = 91 if family == "E3" else 9
        components = []
        assessment_components = []
        for band_index, band in enumerate(BAND_ORDER):
            start = band_index * width
            tr, ass = transform_block(X_raw[train_idx, start:start + width], X_raw[assessment_idx, start:start + width], 3, f"{family}_{band}")
            components.append(tr)
            assessment_components.append(ass)
        eeg_train = np.column_stack(components)
        eeg_assess = np.column_stack(assessment_components)
    train = np.column_stack([X_m2[train_idx], eeg_train])
    assessment = np.column_stack([X_m2[assessment_idx], eeg_assess])
    return transform_block(train, assessment, None, f"{family}_FINAL")


def fit_predict(train_x: np.ndarray, train_y: np.ndarray, assessment_x: np.ndarray, c_value: float, context: dict[str, Any], failures: list[dict[str, Any]]) -> np.ndarray:
    for max_iter in (1000, 10000):
        try:
            model = LogisticRegression(penalty="l2", solver="lbfgs", C=float(c_value), tol=1e-6, max_iter=max_iter, fit_intercept=True, class_weight=None)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(train_x.astype(np.float64), train_y.astype(int))
            convergence = any(issubclass(item.category, ConvergenceWarning) for item in caught)
            if convergence:
                failures.append({**context, "stage": "logistic_fit", "error_code": "CONVERGENCE_WARNING_RETRY", "message": f"max_iter={max_iter}"})
                if max_iter == 1000:
                    continue
                raise FeasibilityFailure("CONVERGENCE_WARNING_AFTER_RETRY")
            if not np.all(np.isfinite(model.coef_)) or not np.all(np.isfinite(model.intercept_)):
                raise FeasibilityFailure("NONFINITE_LOGISTIC_COEFFICIENT")
            probability = model.predict_proba(assessment_x.astype(np.float64))[:, 1]
            if not np.all(np.isfinite(probability)):
                raise FeasibilityFailure("NONFINITE_LOGISTIC_PROBABILITY")
            return probability
        except FeasibilityFailure as exc:
            failures.append({**context, "stage": "logistic_fit", "error_code": str(exc), "message": str(exc)})
            if max_iter == 10000:
                raise
        except Exception as exc:
            failures.append({**context, "stage": "logistic_fit", "error_code": "LOGISTIC_EXCEPTION", "message": repr(exc)})
            raise FeasibilityFailure("LOGISTIC_EXCEPTION") from exc
    raise FeasibilityFailure("LOGISTIC_FAILURE")


def compare_outer_assignments(ids: list[str], y: np.ndarray, predictions: pd.DataFrame, archived_manifest: pd.DataFrame, primary_module) -> dict[str, Any]:
    folds = primary_module.build_folds(ids, y, "group")
    for outer in folds["outer"]:
        repeat, fold = outer["repeat"], outer["outer_fold"]
        pred_ids = set(predictions.loc[(predictions["repeat"] == repeat) & (predictions["outer_fold"] == fold), "participant_id"].astype(str))
        expected_ids = {ids[i] for i in outer["outer_test"]}
        if pred_ids != expected_ids:
            raise RuntimeError(f"ARCHIVED_PRIMARY_OUTER_PREDICTION_ASSIGNMENT_MISMATCH|repeat={repeat}|fold={fold}")
        sub = archived_manifest.loc[(archived_manifest["level"] == "outer") & (archived_manifest["repeat"] == repeat) & (archived_manifest["outer_fold"] == fold)]
        archived_test = set(sub.loc[sub["role"] == "test", "participant_id"].astype(str))
        if archived_test != expected_ids:
            raise RuntimeError(f"ARCHIVED_PRIMARY_FOLD_MANIFEST_ASSIGNMENT_MISMATCH|repeat={repeat}|fold={fold}")
    return folds


def run_family(family: str, phase: dict[str, Any], primary_module, predictions: pd.DataFrame, folds: dict[str, Any], failures: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    ids = phase["ids"]
    X_m2 = pd.read_csv(STAGE1_MATRIX, dtype={"participant_id": str}).set_index("participant_id").loc[ids, M2_COLUMNS].to_numpy(dtype=float)
    X_raw = phase[f"{family.lower()}_frame"].set_index("participant_id").loc[ids].to_numpy(dtype=float)
    outer_predictions: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    family_aborted = False
    for outer in folds["outer"]:
        scores: dict[float, float] = {}
        valid: dict[float, bool] = {}
        for c_value in C_GRID:
            losses: list[float] = []
            candidate_valid = True
            for inner in outer["inner_folds"]:
                context = {"outcome": "group", "family": family, "repeat": outer["repeat"], "outer_fold": outer["outer_fold"], "inner_fold": inner["inner_fold"], "C": c_value}
                try:
                    train_x, valid_x = model_features(X_m2, X_raw, inner["inner_train"], inner["inner_valid"], family)
                    p = fit_predict(train_x, folds["y"][inner["inner_train"]], valid_x, c_value, context, failures)
                    losses.extend(clipped_log_loss(folds["y"][inner["inner_valid"]], p).tolist())
                except Exception as exc:
                    candidate_valid = False
                    failures.append({**context, "stage": "inner_candidate", "error_code": "INVALID_CANDIDATE", "message": str(exc)})
                    break
            if candidate_valid:
                scores[c_value] = float(np.mean(losses))
                valid[c_value] = True
            else:
                scores[c_value] = float("inf")
                valid[c_value] = False
        valid_candidates = [c for c in C_GRID if valid.get(c, False)]
        for c_value in C_GRID:
            selection_rows.append({"family": family, "repeat": outer["repeat"], "outer_fold": outer["outer_fold"], "C": c_value, "pooled_inner_logloss": scores[c_value], "valid": valid.get(c_value, False), "selected": False})
        if not valid_candidates:
            failures.append({"phase": "B", "family": family, "repeat": outer["repeat"], "outer_fold": outer["outer_fold"], "stage": "outer_selection", "error_code": "ALL_C_CANDIDATES_FAILED", "message": "All fixed C candidates invalid for outer fit"})
            family_aborted = True
            break
        selected_c = min(valid_candidates, key=lambda c: (round(scores[c], 12), c))
        selection_rows[-len(C_GRID) + C_GRID.index(selected_c)]["selected"] = True
        context = {"outcome": "group", "family": family, "repeat": outer["repeat"], "outer_fold": outer["outer_fold"], "C": selected_c}
        try:
            train_x, test_x = model_features(X_m2, X_raw, outer["outer_train"], outer["outer_test"], family)
            p_eeg = fit_predict(train_x, folds["y"][outer["outer_train"]], test_x, selected_c, context, failures)
        except Exception as exc:
            failures.append({**context, "stage": "outer_refit", "error_code": "OUTER_REFIT_FAILURE", "message": str(exc)})
            family_aborted = True
            break
        for index, probability in zip(outer["outer_test"], p_eeg):
            row = predictions.loc[(predictions["repeat"] == outer["repeat"]) & (predictions["outer_fold"] == outer["outer_fold"]) & (predictions["participant_id"].astype(str) == ids[index])]
            if len(row) != 1:
                raise RuntimeError(f"ARCHIVED_M2_BASELINE_ROW_FAILURE|{family}|{ids[index]}|{outer['repeat']}|{outer['outer_fold']}")
            row = row.iloc[0]
            outer_predictions.append({"family": family, "repeat": outer["repeat"], "outer_fold": outer["outer_fold"], "participant_index": index, "participant_id": ids[index], "y": int(folds["y"][index]), "p_m2": float(row["p_m2"]), "p_eeg": float(probability), "selected_C": selected_c})
        test_rows = pd.DataFrame([r for r in outer_predictions if r["repeat"] == outer["repeat"] and r["outer_fold"] == outer["outer_fold"]])
        y_test = test_rows["y"].to_numpy(int)
        p2 = test_rows["p_m2"].to_numpy(float)
        pe = test_rows["p_eeg"].to_numpy(float)
        fold_rows.append({
            "family": family, "repeat": outer["repeat"], "outer_fold": outer["outer_fold"], "selected_C": selected_c, "n_test": len(test_rows),
            "m2_logloss": float(clipped_log_loss(y_test, p2).mean()), "eeg_logloss": float(clipped_log_loss(y_test, pe).mean()), "delta_ll": float(np.mean(clipped_log_loss(y_test, p2) - clipped_log_loss(y_test, pe))),
            "m2_auroc": float(roc_auc_score(y_test, p2)), "eeg_auroc": float(roc_auc_score(y_test, pe)), "delta_auroc": float(roc_auc_score(y_test, pe) - roc_auc_score(y_test, p2)),
            "m2_brier": float(np.mean((y_test - p2) ** 2)), "eeg_brier": float(np.mean((y_test - pe) ** 2)), "delta_brier": float(np.mean((y_test - p2) ** 2 - (y_test - pe) ** 2)), "status": "PASS",
        })
    if family_aborted:
        return [], fold_rows, selection_rows
    pred_frame = pd.DataFrame(outer_predictions)
    repeat_rows = []
    for repeat in range(10):
        sub = pred_frame.loc[pred_frame["repeat"] == repeat]
        y_rep = sub["y"].to_numpy(int)
        p2 = sub["p_m2"].to_numpy(float)
        pe = sub["p_eeg"].to_numpy(float)
        repeat_rows.append({
            "family": family, "repeat": repeat, "n": len(sub),
            "m2_logloss": float(clipped_log_loss(y_rep, p2).mean()), "eeg_logloss": float(clipped_log_loss(y_rep, pe).mean()), "delta_ll": float(np.mean(clipped_log_loss(y_rep, p2) - clipped_log_loss(y_rep, pe))),
            "m2_auroc": float(roc_auc_score(y_rep, p2)), "eeg_auroc": float(roc_auc_score(y_rep, pe)), "delta_auroc": float(roc_auc_score(y_rep, pe) - roc_auc_score(y_rep, p2)),
            "m2_brier": float(np.mean((y_rep - p2) ** 2)), "eeg_brier": float(np.mean((y_rep - pe) ** 2)), "delta_brier": float(np.mean((y_rep - p2) ** 2 - (y_rep - pe) ** 2)),
        })
    return outer_predictions, fold_rows, selection_rows


def summarize_repeats(rows: list[dict[str, Any]], key: str = "delta_ll") -> dict[str, float]:
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    return {"mean": float(values.mean()), "median": float(np.median(values)), "iqr_type7": float(np.quantile(values, 0.75, method="linear") - np.quantile(values, 0.25, method="linear")), "min": float(values.min()), "max": float(values.max())}








