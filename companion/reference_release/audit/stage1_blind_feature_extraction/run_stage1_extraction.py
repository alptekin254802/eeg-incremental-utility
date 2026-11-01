"""Outcome-blind Stage-1 extraction for the BALLADEER Attention Robots cohort.

This script is intentionally self-contained and reads only measurement fields:
Measurement screening measurement audit tables, raw Robots EEG/eye/GAME_DATA files, and the
age/source-gender fields from the demographics JSON. It never parses clinical
metadata. All generated files are written below the caller-provided Stage-1
output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import scipy
from scipy import signal


EEG_CHANNELS = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8",
    "FC6", "F4", "F8", "AF4",
]
REGIONS = {
    "frontal": ["AF3", "F7", "F3", "FC5", "FC6", "F4", "F8", "AF4"],
    "temporal": ["T7", "T8"],
    "posterior": ["P7", "O1", "O2", "P8"],
}
BEHAVIOR_FIELDS = [
    "velocidadTrabajoBloque1", "omisionBloque1", "comisionBloque1",
    "velocidadTrabajoBloque2", "omisionBloque2", "comisionBloque2",
    "velocidadTrabajoBloque3", "omisionBloque3", "comisionBloque3",
    "velocidadTrabajoBloque4", "omisionBloque4", "comisionBloque4",
]
FEATURE_COLUMNS = [
    "participant_id", "age", "gender", "mean_work_speed", "total_omissions",
    "total_commissions", "gaze_location_entropy", "gaze_spatial_dispersion",
    "frontal_log_broadband", "temporal_log_broadband", "posterior_log_broadband",
    "frontal_rel_theta", "temporal_rel_theta", "posterior_rel_theta",
    "frontal_rel_alpha", "temporal_rel_alpha", "posterior_rel_alpha",
    "frontal_rel_beta", "temporal_rel_beta", "posterior_rel_beta",
]
EEG_FEATURES = FEATURE_COLUMNS[8:]

# The Measurement screening quality script defines timestamp gaps as > max(0.25, 2.5/fs).
# Reusing that definition keeps cohort reconstruction and filtering boundaries
# aligned without introducing a new data-dependent gap rule.
WINDOW_SECONDS = 2.0
WINDOW_OVERLAP = 0.5
TARGET_FS = 128.0
WINDOW_SAMPLES = int(WINDOW_SECONDS * TARGET_FS)
WINDOW_STEP = int(WINDOW_SAMPLES * (1.0 - WINDOW_OVERLAP))
ABSOLUTE_PTP_UV = 500.0
ROBUST_MAD_Z = 8.0
BAD_POWER_MAD_Z = 5.0
MAX_BAD_CHANNELS = 2
MIN_CLEAN_SECONDS = 120.0
MIN_CLEAN_WINDOW_FRACTION = 0.50
MIN_USABLE_CHANNELS = 12

# Fixed standard 10-20 layout used only for deterministic inverse-distance
# interpolation. These are arbitrary Cartesian coordinates in the scalp plane;
# no coordinate is estimated from the data.
MONTAGE_XY = {
    "AF3": (-0.35, 0.82), "F7": (-0.85, 0.38), "F3": (-0.35, 0.45),
    "FC5": (-0.62, 0.08), "T7": (-1.00, 0.00), "P7": (-0.85, -0.38),
    "O1": (-0.35, -0.82), "O2": (0.35, -0.82), "P8": (0.85, -0.38),
    "T8": (1.00, 0.00), "FC6": (0.62, 0.08), "F4": (0.35, 0.45),
    "F8": (0.85, 0.38), "AF4": (0.35, 0.82),
}

SAFE_EEG_AUDIT_COLS = [
    "participant_id", "session_id", "candidate_complete_modalities", "source_file",
    "file_readable", "device_model", "headset_type", "headset_serial",
    "headset_firmware", "sampling_rate_hz", "sampling_rate_metadata", "channel_count",
    "channel_names", "duration_seconds", "start_timestamp", "stop_timestamp",
    "metadata_samples", "row_count", "timestamp_count", "counter_count",
    "interpolated_row_count", "missing_sample_rows", "missing_channel_cells",
    "expected_channel_cells", "nonfinite_proportion", "duplicate_timestamp_count",
    "nonmonotonic_timestamp_count", "timestamp_gap_count", "longest_timestamp_gap_seconds",
    "counter_duplicate_or_nonmonotonic_count", "flat_channel_count", "flat_channel_names",
    "malformed_row_count", "structurally_usable",
]
SAFE_EYE_AUDIT_COLS = [
    "participant_id", "session_id", "candidate_complete_modalities", "source_file",
    "file_readable", "field_names", "required_fields_present", "missing_required_fields",
    "row_count", "malformed_row_count", "valid_gaze_rows", "invalid_time_rows",
    "invalid_gaze_rows", "valid_gaze_coverage", "missing_gaze_proportion",
    "duration_seconds", "valid_time_span_seconds", "relative_start_seconds",
    "relative_end_seconds", "valid_relative_start_seconds", "valid_relative_end_seconds",
    "median_sample_interval_seconds", "nominal_sampling_hz", "expected_nominal_samples",
    "observed_to_expected_sample_ratio", "longest_missing_gap_seconds", "discontinuity_count",
    "duplicate_timestamp_count", "nonmonotonic_timestamp_count", "looked_col_min",
    "looked_col_max", "looked_row_min", "looked_row_max", "implausible_coordinate_rows",
    "task_coverage_class", "structurally_usable",
]
SAFE_BEHAVIOR_AUDIT_COLS = [
    "participant_id", "session_id", "candidate_complete_modalities", "source_file",
    "file_readable", "field_names", "record_count", "number_task_events",
    "timestamp_fields", "timestamp_completeness", "duration_seconds",
    "task_progression_reconstructable", "structurally_usable",
]


def finite_float(value: Any) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def scalar_from_json_token(token: str) -> Any:
    token = token.strip()
    if token == "null":
        return None
    if token.startswith('"'):
        return json.loads(token)
    number = finite_float(token)
    return number if number is not None else token


def load_allowed_demographics(path: Path) -> dict[str, dict[str, Any]]:
    """Extract only the allowed linkage/demographic keys by targeted matching."""
    text = path.read_text(encoding="utf-8")
    scalar = r'"(?:\\.|[^"\\])*"|null|-?\d+(?:\.\d+)?'

    def values_for(key: str) -> list[Any]:
        matches = re.findall(rf'"{re.escape(key)}"\s*:\s*({scalar})', text)
        return [scalar_from_json_token(m) for m in matches]

    users = values_for("user")
    ages = values_for("age")
    genders = values_for("gender")
    if not (len(users) == len(ages) == len(genders)):
        raise RuntimeError("Allowed demographic fields have inconsistent record counts")
    result: dict[str, dict[str, Any]] = {}
    for user, age, gender in zip(users, ages, genders):
        if not isinstance(user, str) or user in result:
            raise RuntimeError("Demographic linkage key is missing or duplicated")
        result[user] = {"age": finite_float(age), "gender": gender}
    return result


def read_audit(path: Path, usecols: list[str]) -> pd.DataFrame:
    return pd.read_csv(path, usecols=usecols, dtype=str, keep_default_na=False)


def reconstruct_cohort(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audit_dir = root / "audit" / "cohort_screen"
    eeg = read_audit(audit_dir / "robots_eeg_quality.csv", SAFE_EEG_AUDIT_COLS)
    eye = read_audit(audit_dir / "robots_eye_quality.csv", SAFE_EYE_AUDIT_COLS)
    behavior = read_audit(audit_dir / "robots_behavior_quality.csv", SAFE_BEHAVIOR_AUDIT_COLS)

    def num(frame: pd.DataFrame, col: str) -> pd.Series:
        return pd.to_numeric(frame[col], errors="coerce")

    eeg_ok = (
        eeg["candidate_complete_modalities"].map(as_bool)
        & eeg["file_readable"].map(as_bool)
        & eeg["structurally_usable"].map(as_bool)
        & (num(eeg, "duration_seconds") >= 240.0)
        & (num(eeg, "nonfinite_proportion") <= 0.05)
        & (num(eeg, "longest_timestamp_gap_seconds") <= 2.0)
        & (num(eeg, "duplicate_timestamp_count") == 0)
        & (num(eeg, "nonmonotonic_timestamp_count") == 0)
        & (num(eeg, "flat_channel_count") == 0)
    )
    eye_ok = (
        eye["candidate_complete_modalities"].map(as_bool)
        & eye["file_readable"].map(as_bool)
        & eye["structurally_usable"].map(as_bool)
        & (num(eye, "duration_seconds") >= 240.0)
        & (num(eye, "valid_gaze_coverage") >= 0.80)
        & (num(eye, "duplicate_timestamp_count") == 0)
        & (num(eye, "nonmonotonic_timestamp_count") == 0)
    )
    behavior_ok = (
        behavior["candidate_complete_modalities"].map(as_bool)
        & behavior["file_readable"].map(as_bool)
        & behavior["structurally_usable"].map(as_bool)
    )

    def key_frame(frame: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
        result = {}
        for _, row in frame.iterrows():
            key = (row["participant_id"], row["session_id"])
            if key in result:
                raise RuntimeError(f"Duplicate audit record for {key}")
            result[key] = row.to_dict()
        return result

    eeg_map = key_frame(eeg)
    eye_map = key_frame(eye)
    behavior_map = key_frame(behavior)
    keys = sorted(set(eeg_map) & set(eye_map) & set(behavior_map))
    selected_keys = [
        key for i, key in enumerate(keys)
        if bool(eeg_ok.iloc[eeg.index[eeg["participant_id"].eq(key[0]) & eeg["session_id"].eq(key[1])][0]])
        and bool(eye_ok.iloc[eye.index[eye["participant_id"].eq(key[0]) & eye["session_id"].eq(key[1])][0]])
        and bool(behavior_ok.iloc[behavior.index[behavior["participant_id"].eq(key[0]) & behavior["session_id"].eq(key[1])][0]])
    ]
    if len(selected_keys) != 98:
        raise RuntimeError(f"Frozen Measurement screening cohort reconstruction returned {len(selected_keys)}, expected 98")
    participants = [key[0] for key in selected_keys]
    if len(set(participants)) != len(participants):
        raise RuntimeError("Selected Measurement screening cohort contains duplicate participant sessions")

    # Verify raw source existence and record only measurement metadata.
    rows: list[dict[str, Any]] = []
    for participant_id, session_id in selected_keys:
        erow, irow, brow = eeg_map[(participant_id, session_id)], eye_map[(participant_id, session_id)], behavior_map[(participant_id, session_id)]
        eeg_path = root / erow["source_file"]
        eye_path = root / irow["source_file"]
        behavior_path = root / brow["source_file"]
        if not (eeg_path.is_file() and eye_path.is_file() and behavior_path.is_file()):
            raise RuntimeError(f"Required source file missing for {participant_id}/{session_id}")
        rows.append({
            "participant_id": participant_id, "session_id": session_id,
            "eeg_path": eeg_path, "eye_path": eye_path, "behavior_path": behavior_path,
            "eeg_source_file": erow["source_file"], "eye_source_file": irow["source_file"],
            "behavior_source_file": brow["source_file"],
        })
    summary = {
        "measurement_selected_n": len(rows),
        "unique_participants": len(set(participants)),
        "duplicate_participant_session_records": len(selected_keys) - len(set(selected_keys)),
        "all_required_source_files_exist": True,
        "measurement_eeg_duration_rule_seconds": 240.0,
        "measurement_eeg_nonfinite_rule": 0.05,
        "measurement_eeg_timestamp_gap_rule_seconds": 2.0,
        "measurement_eye_duration_rule_seconds": 240.0,
        "measurement_eye_valid_coverage_rule": 0.80,
        "measurement_eye_longest_gap_rule_applied": False,
    }
    return rows, summary


def parse_metadata_line(line: str) -> dict[str, Any]:
    fields = {
        "participant_id": r"title:([^,]+)",
        "start_timestamp": r"start timestamp:([^,]+)",
        "stop_timestamp": r"stop timestamp:([^,]+)",
        "headset_type": r"headset type:([^,]+)",
        "headset_serial": r"headset serial:([^,]+)",
        "headset_firmware": r"headset firmware:([^,]+)",
        "sampling_rate_metadata": r"sampling rate:([^,]+)",
        "metadata_samples": r"samples:\s*([^,]+)",
        "metadata_version": r"version:([^,]+)",
    }
    out: dict[str, Any] = {}
    for key, pattern in fields.items():
        match = re.search(pattern, line, flags=re.IGNORECASE)
        value = match.group(1).strip() if match else ""
        if key in {"start_timestamp", "stop_timestamp"}:
            out[key] = finite_float(value)
        elif key == "metadata_samples":
            number = finite_float(value)
            out[key] = int(number) if number is not None else None
        else:
            out[key] = value
    rate_match = re.search(r"eeg_(\d+(?:\.\d+)?)", str(out.get("sampling_rate_metadata", "")), re.I)
    out["sampling_rate_hz"] = finite_float(rate_match.group(1)) if rate_match else None
    out["device_model"] = {
        "EPOCX": "Epoc X", "EPOCPLUS": "Epoc+",
    }.get(str(out.get("headset_type", "")).strip().upper(), str(out.get("headset_type", "")).strip())
    return out


def parse_number(value: str) -> float:
    number = finite_float(value)
    return np.nan if number is None else number


def parse_eeg(path: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        metadata_line = handle.readline()
        header_line = handle.readline()
        if not metadata_line or not header_line:
            raise RuntimeError("EEG file has fewer than two header lines")
        metadata = parse_metadata_line(metadata_line)
        header = next(csv.reader([header_line]))
        header = [name.strip() for name in header]
        index = {name: i for i, name in enumerate(header)}
        required = ["Timestamp", "EEG.Counter", "EEG.Interpolated"] + [f"EEG.{c}" for c in EEG_CHANNELS]
        missing = [name for name in required if name not in index]
        if missing:
            raise RuntimeError("Missing required EEG columns: " + ",".join(missing))
        values: list[list[float]] = []
        timestamps: list[float] = []
        counters: list[float] = []
        interpolation_flags: list[bool] = []
        malformed_rows = 0
        for raw in csv.reader(handle):
            if not raw or all(not str(value).strip() for value in raw):
                continue
            if len(raw) < max(index[name] for name in required) + 1:
                malformed_rows += 1
            timestamps.append(parse_number(raw[index["Timestamp"]]) if len(raw) > index["Timestamp"] else np.nan)
            counters.append(parse_number(raw[index["EEG.Counter"]]) if len(raw) > index["EEG.Counter"] else np.nan)
            interp = parse_number(raw[index["EEG.Interpolated"]]) if len(raw) > index["EEG.Interpolated"] else np.nan
            interpolation_flags.append(bool(math.isfinite(interp) and interp > 0))
            values.append([
                parse_number(raw[index[f"EEG.{channel}"]]) if len(raw) > index[f"EEG.{channel}"] else np.nan
                for channel in EEG_CHANNELS
            ])
    data = np.asarray(values, dtype=np.float64)
    timestamps_a = np.asarray(timestamps, dtype=np.float64)
    counters_a = np.asarray(counters, dtype=np.float64)
    interp_a = np.asarray(interpolation_flags, dtype=bool)
    if data.ndim != 2 or data.shape[1] != len(EEG_CHANNELS):
        raise RuntimeError("Parsed EEG signal is not a 14-channel matrix")
    finite_t = timestamps_a[np.isfinite(timestamps_a)]
    deltas = np.diff(finite_t)
    rate = float(metadata["sampling_rate_hz"] or 0.0)
    gap_limit = max(0.25, 2.5 / rate) if rate > 0 else np.nan
    gaps = deltas[deltas > gap_limit] if math.isfinite(gap_limit) else np.array([], dtype=float)
    metadata.update({
        "sample_count": int(data.shape[0]),
        "timestamp_count": int(np.isfinite(timestamps_a).sum()),
        "counter_count": int(np.isfinite(counters_a).sum()),
        "interpolated_row_count": int(interp_a.sum()),
        "malformed_row_count": int(malformed_rows),
        "nonfinite_channel_cells": int((~np.isfinite(data)).sum()),
        "nonfinite_proportion": float((~np.isfinite(data)).mean()),
        "duplicate_timestamp_count": int((deltas == 0).sum()),
        "nonmonotonic_timestamp_count": int((deltas < 0).sum()),
        "timestamp_gap_count": int(gaps.size),
        "longest_timestamp_gap_seconds": float(gaps.max()) if gaps.size else 0.0,
        "counter_duplicate_or_nonmonotonic_count": int(np.sum(np.diff(counters_a[np.isfinite(counters_a)]) <= 0)) if np.isfinite(counters_a).sum() > 1 else 0,
        "original_duration_seconds": float((metadata["stop_timestamp"] - metadata["start_timestamp"]) if metadata["start_timestamp"] is not None and metadata["stop_timestamp"] is not None else np.nan),
        "channel_count": len(EEG_CHANNELS),
        "channel_names": ";".join(EEG_CHANNELS),
        "structural_parser_status": "PASS" if malformed_rows == 0 and np.isfinite(timestamps_a).all() else "FAIL_STRUCTURAL",
    })
    return metadata, data, timestamps_a, counters_a, interp_a, gaps


def max_exact_run(values: np.ndarray, valid: np.ndarray) -> int:
    max_run = run = 0
    previous = None
    for value, is_valid in zip(values, valid):
        if not is_valid:
            run = 0
            previous = None
            continue
        if run and value == previous:
            run += 1
        else:
            run = 1
        previous = value
        max_run = max(max_run, run)
    return max_run


def welch_band_power(window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frequencies, psd = signal.welch(
        window, fs=TARGET_FS, window="hann", nperseg=WINDOW_SAMPLES,
        noverlap=0, detrend="constant", scaling="density", average="mean", axis=0,
    )
    return frequencies, psd


def integrate_psd(frequencies: np.ndarray, values: np.ndarray, low: float, high: float) -> float:
    mask = (frequencies >= low) & (frequencies <= high)
    if mask.sum() < 2:
        return float("nan")
    return float(np.trapezoid(values[mask], frequencies[mask]))


def robust_threshold(values: np.ndarray, z: float) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    scale = 1.4826 * mad
    return med + z * scale if scale > 0 else med


def bad_channels_from_raw(data: np.ndarray, rate: float) -> tuple[list[int], dict[str, Any]]:
    finite = np.isfinite(data)
    nonfinite_fraction = (~finite).mean(axis=0)
    flat_limit = int(math.floor(5.0 * rate)) + 1
    flat_runs = np.array([max_exact_run(data[:, i], finite[:, i]) for i in range(data.shape[1])])

    # A deterministic broadband diagnostic for the within-participant power rule:
    # Welch on each channel's longest finite contiguous segment, then integrate 1-40 Hz.
    log_power = np.full(data.shape[1], np.nan, dtype=float)
    nperseg = min(WINDOW_SAMPLES, max(8, int(round(rate * 2.0))))
    for i in range(data.shape[1]):
        good = finite[:, i]
        segments = contiguous_true_runs(good)
        if not segments:
            continue
        start, stop = max(segments, key=lambda pair: pair[1] - pair[0])
        segment = data[start:stop, i]
        if segment.size < nperseg:
            continue
        f, p = signal.welch(segment, fs=rate, window="hann", nperseg=nperseg,
                             noverlap=0, detrend="constant", scaling="density")
        power = integrate_psd(f, p, 1.0, min(40.0, rate / 2.0 - 1e-9))
        if power > 0 and math.isfinite(power):
            log_power[i] = math.log10(power)
    finite_lp = np.isfinite(log_power)
    lp_median = float(np.median(log_power[finite_lp])) if finite_lp.any() else np.nan
    lp_mad = float(np.median(np.abs(log_power[finite_lp] - lp_median))) if finite_lp.any() else np.nan
    lp_scale = 1.4826 * lp_mad if math.isfinite(lp_mad) else np.nan
    if finite_lp.any() and math.isfinite(lp_scale) and lp_scale > 0:
        power_bad = np.abs(log_power - lp_median) > BAD_POWER_MAD_Z * lp_scale
    elif finite_lp.any():
        power_bad = finite_lp & (log_power != lp_median)
    else:
        power_bad = np.ones(data.shape[1], dtype=bool)
    bad = (~finite_fraction_ok(nonfinite_fraction, 0.05)) | (flat_runs > flat_limit) | power_bad
    details = {
        "nonfinite_fraction_by_channel": nonfinite_fraction.tolist(),
        "flatline_max_run_samples_by_channel": flat_runs.tolist(),
        "flatline_limit_samples": flat_limit,
        "log_broadband_power_by_channel": log_power.tolist(),
        "log_broadband_median": lp_median,
        "log_broadband_mad": lp_mad,
        "bad_power_threshold": (lp_median + BAD_POWER_MAD_Z * lp_scale) if finite_lp.any() and math.isfinite(lp_scale) else np.nan,
    }
    return [int(i) for i in np.flatnonzero(bad)], details


def finite_fraction_ok(values: np.ndarray, limit: float) -> np.ndarray:
    return values <= limit


def contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    padded = np.concatenate(([False], mask, [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    stops = np.flatnonzero(padded[:-1] & ~padded[1:])
    return [(int(start), int(stop)) for start, stop in zip(starts, stops) if stop > start]


def interpolate_bad_channels(data: np.ndarray, bad: list[int]) -> np.ndarray:
    out = data.copy()
    good = [i for i in range(data.shape[1]) if i not in bad]
    for bad_i in bad:
        target = np.asarray(MONTAGE_XY[EEG_CHANNELS[bad_i]], dtype=float)
        distances = np.asarray([
            np.linalg.norm(target - np.asarray(MONTAGE_XY[EEG_CHANNELS[i]], dtype=float))
            for i in good
        ])
        weights = 1.0 / np.maximum(distances, 1e-12) ** 2
        weights /= weights.sum()
        out[:, bad_i] = np.sum(out[:, good] * weights[None, :], axis=1)
    return out


def segment_bounds(timestamps: np.ndarray, invalid_rows: np.ndarray, rate: float) -> list[tuple[int, int]]:
    good = np.isfinite(timestamps) & ~invalid_rows
    gap_limit = max(0.25, 2.5 / rate)
    output: list[tuple[int, int]] = []
    for start, stop in contiguous_true_runs(good):
        local_t = timestamps[start:stop]
        local_deltas = np.diff(local_t)
        split_after = np.flatnonzero((local_deltas <= 0) | (local_deltas > gap_limit))
        left = start
        for relative_index in split_after:
            right = start + int(relative_index) + 1
            if right > left:
                output.append((left, right))
            left = right
        if stop > left:
            output.append((left, stop))
    return output


def resample_and_preprocess(data: np.ndarray, timestamps: np.ndarray, interp: np.ndarray,
                            bad: list[int], rate: float) -> tuple[list[np.ndarray], list[tuple[int, int]], dict[str, Any]]:
    invalid_rows = (~np.isfinite(data).all(axis=1)) | interp | (~np.isfinite(timestamps))
    bounds = segment_bounds(timestamps, invalid_rows, rate)
    processed_segments: list[np.ndarray] = []
    processed_bounds: list[tuple[int, int]] = []
    sos = signal.butter(4, [1.0, 40.0], btype="bandpass", fs=TARGET_FS, output="sos")
    for start, stop in bounds:
        segment = data[start:stop].copy()
        if segment.shape[0] < max(WINDOW_SAMPLES, 32):
            continue
        segment = interpolate_bad_channels(segment, bad)
        if rate == 256.0:
            segment = signal.resample_poly(segment, up=1, down=2, axis=0,
                                           window=("kaiser", 5.0), padtype="constant")
        elif rate != TARGET_FS:
            raise RuntimeError(f"Unsupported native sampling rate: {rate}")
        if segment.shape[0] < WINDOW_SAMPLES:
            continue
        try:
            segment = signal.sosfiltfilt(sos, segment, axis=0)
        except ValueError:
            continue
        segment = segment - np.mean(segment, axis=1, keepdims=True)
        processed_segments.append(segment)
        processed_bounds.append((start, stop))
    return processed_segments, processed_bounds, {
        "invalid_row_count": int(invalid_rows.sum()),
        "invalid_row_fraction": float(invalid_rows.mean()) if invalid_rows.size else 1.0,
        "segment_count": len(processed_segments),
        "filter": "scipy.signal.butter(order=4, btype='bandpass', Wn=[1,40], fs=128, output='sos') + sosfiltfilt",
        "resampling": "scipy.signal.resample_poly(up=1, down=2, window=('kaiser',5.0), padtype='constant') for native 256 Hz",
    }


def artifact_and_features(segments: list[np.ndarray]) -> tuple[dict[str, float], dict[str, Any]]:
    windows: list[np.ndarray] = []
    window_ids: list[tuple[int, int]] = []
    for segment_i, segment in enumerate(segments):
        for start in range(0, segment.shape[0] - WINDOW_SAMPLES + 1, WINDOW_STEP):
            windows.append(segment[start:start + WINDOW_SAMPLES])
            window_ids.append((segment_i, start))
    if not windows:
        return {}, {"candidate_window_count": 0, "rejected_window_count": 0, "retained_window_count": 0,
                    "clean_window_fraction": 0.0, "clean_seconds": 0.0, "artifact_threshold_robust": np.nan}
    ptp = np.asarray([np.ptp(signal.detrend(window, axis=0, type="linear"), axis=0) for window in windows])
    robust_limit = robust_threshold(ptp.ravel(), ROBUST_MAD_Z)
    rejected = (ptp > ABSOLUTE_PTP_UV).any(axis=1) | (ptp > robust_limit).any(axis=1)
    retained = [windows[i] for i in range(len(windows)) if not rejected[i]]
    retained_ids = [window_ids[i] for i in range(len(windows)) if not rejected[i]]
    # Union of retained 2-second intervals on each processed segment. This
    # avoids double-counting the 50%-overlapped windows in clean duration.
    intervals: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for seg_i, start in retained_ids:
        intervals[seg_i].append((start, start + WINDOW_SAMPLES))
    clean_samples = 0
    for seg_i, pairs in intervals.items():
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
        "artifact_absolute_threshold_uv": ABSOLUTE_PTP_UV,
    }
    if not retained:
        return {}, qc
    # Vectorize across windows while retaining the frozen per-window Welch
    # definition. The output is identical to calling welch on each window
    # independently, but avoids a large Python call overhead.
    stacked = np.stack(retained, axis=0)
    frequencies, psd = signal.welch(
        stacked, fs=TARGET_FS, window="hann", nperseg=WINDOW_SAMPLES,
        noverlap=0, detrend="constant", scaling="density", average="mean", axis=1,
    )
    def integrate_axis(low: float, high: float) -> np.ndarray:
        mask = (frequencies >= low) & (frequencies <= high)
        if mask.sum() < 2:
            return np.full((stacked.shape[0], stacked.shape[2]), np.nan)
        return np.trapezoid(psd[:, mask, :], frequencies[mask], axis=1)

    broadband_values = integrate_axis(1.0, 40.0)
    theta_values = integrate_axis(4.0, 7.0)
    alpha_values = integrate_axis(8.0, 12.0)
    beta_values = integrate_axis(13.0, 30.0)
    channel_features: dict[str, dict[str, float]] = {}
    for channel_i, channel in enumerate(EEG_CHANNELS):
        broadband = broadband_values[:, channel_i]
        channel_features[channel] = {
            "log_broadband": float(np.log10(np.nanmedian(broadband))),
            "rel_theta": float(np.nanmedian(theta_values[:, channel_i] / broadband)),
            "rel_alpha": float(np.nanmedian(alpha_values[:, channel_i] / broadband)),
            "rel_beta": float(np.nanmedian(beta_values[:, channel_i] / broadband)),
        }
    features: dict[str, float] = {}
    for region, channels in REGIONS.items():
        for feature_name in ("log_broadband", "rel_theta", "rel_alpha", "rel_beta"):
            values = [channel_features[channel][feature_name] for channel in channels]
            features[f"{region}_{feature_name}"] = float(np.nanmedian(values))
    return features, qc


def extract_behavior(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) != 2:
        raise RuntimeError("GAME_DATA must contain exactly one summary row")
    header = [value.strip() for value in rows[0]]
    row = rows[1]
    index = {name: i for i, name in enumerate(header)}
    missing = [field for field in BEHAVIOR_FIELDS if field not in index]
    if missing:
        raise RuntimeError("Missing frozen behavior fields: " + ",".join(missing))
    values = {field: finite_float(row[index[field]]) if index[field] < len(row) else None for field in BEHAVIOR_FIELDS}
    if any(value is None for value in values.values()):
        raise RuntimeError("Nonfinite frozen behavior value")
    speeds = [values[f"velocidadTrabajoBloque{i}"] for i in range(1, 5)]
    omissions = [values[f"omisionBloque{i}"] for i in range(1, 5)]
    commissions = [values[f"comisionBloque{i}"] for i in range(1, 5)]
    return {
        "mean_work_speed": float(np.mean(speeds)),
        "total_omissions": float(np.sum(omissions)),
        "total_commissions": float(np.sum(commissions)),
    }


def run_behavior_formula_tests(root: Path) -> list[dict[str, Any]]:
    # Values below were manually inspected in the corresponding raw summary
    # row before execution; this tests the direct frozen formulas, not labels.
    fixture = {
        "path": "dataset/UB0004/AttentionRobotsDesktop/1686235332/UB0004_GAME_DATA_2023_06_08T16.36.48+02.00.csv",
        "speeds": [20.66667, 23.33333, 23.66667, 23.66667],
        "omissions": [2.0, 2.0, 3.0, 2.0],
        "commissions": [1.0, 0.0, 0.0, 0.0],
        "expected": {"mean_work_speed": 22.833335, "total_omissions": 9.0, "total_commissions": 1.0},
    }
    observed = {
        "mean_work_speed": float(np.mean(fixture["speeds"])),
        "total_omissions": float(np.sum(fixture["omissions"])),
        "total_commissions": float(np.sum(fixture["commissions"])),
    }
    for key, expected in fixture["expected"].items():
        if not math.isclose(observed[key], expected, rel_tol=0.0, abs_tol=1e-8):
            raise AssertionError(f"Behavior formula test failed for {key}")
    actual = extract_behavior(root / fixture["path"])
    for key, expected in fixture["expected"].items():
        if not math.isclose(actual[key], expected, rel_tol=0.0, abs_tol=1e-6):
            raise AssertionError(f"Raw behavior formula test failed for {key}")
    return [{"fixture": fixture["path"], "status": "PASS", **observed}]


def extract_gaze(path: Path) -> dict[str, float]:
    counts: Counter[tuple[int, int]] = Counter()
    points: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"timeChecked", "looked_col", "looked_row"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise RuntimeError("Gaze file lacks required object-location fields")
        for row in reader:
            col = finite_float(row.get("looked_col"))
            row_value = finite_float(row.get("looked_row"))
            time_value = finite_float(row.get("timeChecked"))
            if col is None or row_value is None or time_value is None:
                continue
            if not (1 <= col <= 47 and 1 <= row_value <= 14):
                continue
            col_i, row_i = int(round(col)), int(round(row_value))
            counts[(row_i, col_i)] += 1
            points.append(((col_i - 1) / 46.0, (row_i - 1) / 13.0))
    if not points:
        raise RuntimeError("No valid object-location rows")
    total = float(sum(counts.values()))
    probabilities = np.asarray([count / total for count in counts.values()], dtype=float)
    entropy = float(-np.sum(probabilities * np.log(probabilities)) / math.log(47.0 * 14.0))
    point_array = np.asarray(points, dtype=float)
    centroid = point_array.mean(axis=0)
    dispersion = float(np.sqrt(np.mean(np.sum((point_array - centroid) ** 2, axis=1))))
    return {"gaze_location_entropy": entropy, "gaze_spatial_dispersion": dispersion}


def parser_audit_row(record: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "participant_id": record["participant_id"], "session_id": record["session_id"],
        "source_file": record["eeg_source_file"], "device": metadata["device_model"],
        "headset_type": metadata["headset_type"], "headset_serial": metadata["headset_serial"],
        "headset_firmware": metadata["headset_firmware"], "native_sampling_rate_hz": metadata["sampling_rate_hz"],
        "sampling_rate_metadata": metadata["sampling_rate_metadata"],
        "original_duration_seconds": metadata["original_duration_seconds"],
        "metadata_sample_count": metadata["metadata_samples"], "sample_count": metadata["sample_count"],
        "timestamp_count": metadata["timestamp_count"], "counter_count": metadata["counter_count"],
        "timestamp_monotonic": metadata["duplicate_timestamp_count"] == 0 and metadata["nonmonotonic_timestamp_count"] == 0,
        "duplicate_timestamp_count": metadata["duplicate_timestamp_count"],
        "nonmonotonic_timestamp_count": metadata["nonmonotonic_timestamp_count"],
        "discontinuity_count": metadata["timestamp_gap_count"],
        "longest_discontinuity_seconds": metadata["longest_timestamp_gap_seconds"],
        "nominal_sampling_rate_hz": metadata["sampling_rate_hz"], "channel_count": metadata["channel_count"],
        "channel_names": metadata["channel_names"], "missing_channel_cells": metadata["nonfinite_channel_cells"],
        "nonfinite_proportion": metadata["nonfinite_proportion"],
        "interpolated_row_count": metadata["interpolated_row_count"],
        "malformed_row_count": metadata["malformed_row_count"],
        "parser_status": metadata["structural_parser_status"], "parser_error": "",
    }


def process_record(record: dict[str, Any], demographics: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    participant = record["participant_id"]
    if participant not in demographics or demographics[participant]["age"] is None:
        raise RuntimeError(f"Allowed demographics missing for {participant}")
    metadata, data, timestamps, counters, interp, _ = parse_eeg(record["eeg_path"])
    parser_row = parser_audit_row(record, metadata)
    bad_indices, bad_details = bad_channels_from_raw(data, float(metadata["sampling_rate_hz"]))
    bad_names = [EEG_CHANNELS[i] for i in bad_indices]
    qc: dict[str, Any] = {
        "participant_id": participant, "session_id": record["session_id"],
        "native_sampling_rate_hz": metadata["sampling_rate_hz"], "device": metadata["device_model"],
        "original_duration_seconds": metadata["original_duration_seconds"],
        "discontinuity_count": metadata["timestamp_gap_count"],
        "bad_channel_count": len(bad_indices), "bad_channel_names": ";".join(bad_names),
        "interpolation_count": len(bad_indices) if len(bad_indices) <= MAX_BAD_CHANNELS else 0,
        "candidate_window_count": 0, "rejected_window_count": 0, "clean_window_fraction": 0.0,
        "clean_seconds": 0.0, "final_usable_channels": max(0, len(EEG_CHANNELS) - len(bad_indices)) if len(bad_indices) > MAX_BAD_CHANNELS else len(EEG_CHANNELS),
        "frontal_usable_channels": sum(c not in bad_names for c in REGIONS["frontal"]),
        "temporal_usable_channels": sum(c not in bad_names for c in REGIONS["temporal"]),
        "posterior_usable_channels": sum(c not in bad_names for c in REGIONS["posterior"]),
        "preprocessing_status": "", "exclusion_reason": "", "interpolation_geometry": "fixed_inverse_distance_squared",
        "absolute_artifact_threshold_applied": True, "absolute_artifact_threshold_uv": ABSOLUTE_PTP_UV,
        "robust_artifact_rule": ">8*1.4826*MAD of participant channel-window peak-to-peak values",
    }
    if metadata["structural_parser_status"] != "PASS" or metadata["channel_count"] != 14 or metadata["sampling_rate_hz"] not in {128.0, 256.0}:
        qc["preprocessing_status"] = "FAIL_STRUCTURAL"
        qc["exclusion_reason"] = "parser, channel, timestamp, or sampling metadata failure"
        return qc, parser_row, {}
    if len(bad_indices) > MAX_BAD_CHANNELS:
        qc["preprocessing_status"] = "FAIL_TOO_MANY_BAD_CHANNELS"
        qc["exclusion_reason"] = ">2 frozen bad channels"
        return qc, parser_row, {}
    segments, _, prep_detail = resample_and_preprocess(data, timestamps, interp, bad_indices, float(metadata["sampling_rate_hz"]))
    eeg_features, artifact_qc = artifact_and_features(segments)
    qc.update(artifact_qc)
    if not segments or qc["candidate_window_count"] == 0:
        qc["preprocessing_status"] = "FAIL_CLEAN_DURATION"
        qc["exclusion_reason"] = "no usable clean 2-second windows"
        return qc, parser_row, {}
    if qc["clean_seconds"] < MIN_CLEAN_SECONDS:
        qc["preprocessing_status"] = "FAIL_CLEAN_DURATION"
        qc["exclusion_reason"] = f"clean_seconds<{MIN_CLEAN_SECONDS}"
    elif qc["clean_window_fraction"] < MIN_CLEAN_WINDOW_FRACTION:
        qc["preprocessing_status"] = "FAIL_CLEAN_WINDOW_FRACTION"
        qc["exclusion_reason"] = f"clean_window_fraction<{MIN_CLEAN_WINDOW_FRACTION}"
    elif qc["final_usable_channels"] < MIN_USABLE_CHANNELS:
        qc["preprocessing_status"] = "FAIL_REGION_COVERAGE"
        qc["exclusion_reason"] = f"final_usable_channels<{MIN_USABLE_CHANNELS}"
    elif any(qc[f"{region}_usable_channels"] < 1 for region in ("frontal", "temporal", "posterior")):
        qc["preprocessing_status"] = "FAIL_REGION_COVERAGE"
        qc["exclusion_reason"] = "missing usable frozen region"
    elif not eeg_features or any(not math.isfinite(value) for value in eeg_features.values()):
        qc["preprocessing_status"] = "OTHER_FROZEN_RULE_FAILURE"
        qc["exclusion_reason"] = "undefined primary EEG feature"
    else:
        qc["preprocessing_status"] = "PASS"
        qc["exclusion_reason"] = ""
    qc["preprocessing_detail"] = json.dumps({**prep_detail, **bad_details}, sort_keys=True, allow_nan=False, default=str)
    return qc, parser_row, eeg_features if qc["preprocessing_status"] == "PASS" else {}


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def numeric_summary(values: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return {"N_defined": 0, "missing_N": int(len(values)), "min": None, "median": None, "max": None, "IQR": None, "zero_variance": None, "finite_nonfinite_status": "no_defined_values"}
    q1, q3 = finite.quantile([0.25, 0.75])
    return {"N_defined": int(finite.size), "missing_N": int(len(values) - finite.size),
            "min": float(finite.min()), "median": float(finite.median()), "max": float(finite.max()),
            "IQR": float(q3 - q1), "zero_variance": bool(finite.nunique() <= 1),
            "finite_nonfinite_status": "all_defined_values_finite" if finite.size == len(values) else "missing_or_nonfinite_values_present"}


def structural_qc(matrix: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries = []
    for feature in FEATURE_COLUMNS[1:]:
        summary = numeric_summary(matrix[feature])
        summary["feature"] = feature
        summaries.append(summary)
    redundancy: list[dict[str, Any]] = []
    predictors = FEATURE_COLUMNS[1:]
    for i, left in enumerate(predictors):
        for right in predictors[i + 1:]:
            x = pd.to_numeric(matrix[left], errors="coerce")
            y = pd.to_numeric(matrix[right], errors="coerce")
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 2 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
                corr = np.nan
            else:
                corr = float(np.corrcoef(x[mask], y[mask])[0, 1])
            exact = bool(mask.all() and np.array_equal(x[mask].to_numpy(), y[mask].to_numpy()))
            near = bool(math.isfinite(corr) and abs(corr) >= 0.999999)
            if exact or near:
                redundancy.append({"feature_a": left, "feature_b": right, "N_pairwise": int(mask.sum()),
                                   "pearson_r": corr, "exact_duplicate": exact, "near_exact_abs_r_ge_0.999999": near,
                                   "decision": "retain_unless_exact_mathematical_duplication_or_implementation_error"})
    # Pooled impossibility checks only; no labels.
    for feature in FEATURE_COLUMNS[1:]:
        values = pd.to_numeric(matrix[feature], errors="coerce")
        if feature == "gaze_location_entropy":
            if ((values < 0) | (values > 1)).any():
                raise AssertionError("Impossible entropy value")
        if feature == "gaze_spatial_dispersion":
            if ((values < 0) | (values > math.sqrt(2))).any():
                raise AssertionError("Impossible spatial-dispersion value")
        if feature in {"mean_work_speed", "total_omissions", "total_commissions"}:
            if (values < 0).any():
                raise AssertionError(f"Impossible negative behavior value in {feature}")
        if feature.startswith("rel_") or "_rel_" in feature:
            if ((values < 0) | (values > 1)).any():
                raise AssertionError(f"Impossible relative-power value in {feature}")
    return summaries, redundancy


















