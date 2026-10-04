"""Outcome-blind Measurement screening structural audit for Attention Robots.

The primary label is fixed to users_demographics.json ``group`` (1 =
Experimental, 2 = Control).  ``diagnosed`` is carried as metadata only.

This script reads only structural metadata, file headers, timestamps, file
presence, and diagnosis-blind quality indicators.  It does not calculate
EEG features, gaze features, behavioural effects, group comparisons, or
classification outcomes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset"
OUT = ROOT / "audit" / "cohort_screen"
METADATA_PATH = DATASET / "users_demographics.json"
README_PATH = DATASET / "README.md"
EMBRACE_PATH = DATASET / "balladeer_embraceplus_data.csv"

GROUP_LABELS = {1: "Experimental", 2: "Control"}
SEX_LABELS = {"1": "Male", "2": "Female", 1: "Male", 2: "Female"}
TASK_NAME = "Attention Robots"

# These values are frozen only after the pooled quality distributions are
# inspected.  They are measurement-validity rules, not outcome-optimized
# selection rules.  A session-level GAME_DATA summary cannot satisfy a strict
# event-level three-way synchronization rule because it has no event clock.
FROZEN_THRESHOLDS = {
    "eeg_min_duration_seconds": 240.0,
    "eeg_max_nonfinite_proportion": 0.05,
    "eeg_max_longest_timestamp_gap_seconds": 2.0,
    "eye_min_duration_seconds": 240.0,
    "eye_min_valid_gaze_coverage": 0.80,
    "eye_max_longest_missing_gap_seconds_sensitivity": 60.0,
    "eye_gap_hard_rule_applied": False,
    "behavior_required_readable_summary": True,
    "strict_multimodal_sync_status": "ALIGNABLE_WITH_OFFSET",
}


def norm(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def finite_float(value: object) -> float | None:
    text = norm(value)
    if not text or text.lower() in {"nan", "na", "n/a", "null", "none", "-"}:
        return None
    try:
        result = float(text)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def bool_text(value: bool | None) -> str:
    if value is None:
        return "unknown"
    return "true" if value else "false"




def csv_float(value: float | None, digits: int = 6) -> object:
    if value is None or not math.isfinite(value):
        return ""
    return round(value, digits)




def read_csv_rows(path: Path, delimiter: str = ",") -> tuple[list[str], list[list[str]], bool, str]:
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            header = next(reader, None)
            if header is None:
                return [], [], True, "empty_file"
            rows = list(reader)
        return header, rows, True, ""
    except Exception as exc:  # pragma: no cover - defensive file audit path
        return [], [], False, f"{type(exc).__name__}:{exc}"








def parse_metadata_line(line: str) -> dict[str, object]:
    result: dict[str, object] = {}
    patterns = {
        "participant_id": r"title:([^,]+)",
        "start_timestamp": r"start timestamp:([^,]+)",
        "stop_timestamp": r"stop timestamp:([^,]+)",
        "headset_type": r"headset type:([^,]+)",
        "headset_serial": r"headset serial:([^,]+)",
        "headset_firmware": r"headset firmware:([^,]+)",
        "metadata_channels": r"channels:([^,]+)",
        "sampling_rate_metadata": r"sampling rate:([^,]+)",
        "metadata_samples": r"samples:\s*([^,]+)",
        "metadata_version": r"version:([^,]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, line, flags=re.IGNORECASE)
        result[key] = norm(match.group(1)) if match else ""
    result["start_timestamp"] = finite_float(result.get("start_timestamp"))
    result["stop_timestamp"] = finite_float(result.get("stop_timestamp"))
    sample_text = norm(result.get("metadata_samples"))
    try:
        result["metadata_samples"] = int(float(sample_text))
    except ValueError:
        result["metadata_samples"] = ""
    return result




def find_files(session_dir: Path) -> dict[str, Path | None]:
    files = [path for path in session_dir.iterdir() if path.is_file()]
    eeg = [path for path in files if "EPOC" in path.name.upper() and path.suffix.lower() == ".csv"]
    eye = [path for path in files if "EYE_TRACKING_DATA" in path.name.upper()]
    game = [path for path in files if "GAME_DATA" in path.name.upper()]
    return {
        "eeg": sorted(eeg)[0] if eeg else None,
        "eye": sorted(eye)[0] if eye else None,
        "behavior": sorted(game)[0] if game else None,
    }




def discover_sessions(meta: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    sessions: list[dict[str, object]] = []
    for participant_dir in sorted(DATASET.iterdir(), key=lambda path: path.name):
        if not participant_dir.is_dir() or participant_dir.name not in meta:
            continue
        robots_dir = participant_dir / "AttentionRobotsDesktop"
        if not robots_dir.is_dir():
            continue
        for session_dir in sorted((path for path in robots_dir.iterdir() if path.is_dir()), key=lambda path: path.name):
            files = find_files(session_dir)
            sessions.append({
                "participant_id": participant_dir.name,
                "session_id": session_dir.name,
                "session_dir": session_dir,
                "eeg_path": files["eeg"],
                "eye_path": files["eye"],
                "behavior_path": files["behavior"],
                "eeg_available": files["eeg"] is not None,
                "eye_available": files["eye"] is not None,
                "behavior_available": files["behavior"] is not None,
            })
    return sessions


def robust_outlier_fraction(values_by_channel: list[list[float]]) -> float:
    total = 0
    outliers = 0
    for values in values_by_channel:
        if not values:
            continue
        median = statistics.median(values)
        deviations = [abs(value - median) for value in values]
        mad = statistics.median(deviations)
        total += len(values)
        if mad > 0:
            limit = 10.0 * 1.4826 * mad
            outliers += sum(1 for value in values if abs(value - median) > limit)
    return outliers / total if total else 1.0


def saturation_fraction(values_by_channel: list[list[float]]) -> float:
    total = 0
    saturated = 0
    for values in values_by_channel:
        if not values:
            continue
        min_value = min(values)
        max_value = max(values)
        total += len(values)
        # This is a structural clipping screen, not a physiological feature.
        min_count = sum(1 for value in values if value == min_value)
        max_count = sum(1 for value in values if value == max_value)
        if min_count / len(values) > 0.01 or max_count / len(values) > 0.01:
            saturated += max(min_count, max_count)
    return saturated / total if total else 1.0


def parse_eeg(path: Path, candidate: bool) -> dict[str, object]:
    row: dict[str, object] = {
        "source_file": str(path.relative_to(ROOT)),
        "candidate_complete_modalities": bool_text(candidate),
        "file_readable": "false",
    }
    timestamps: list[float] = []
    counters: list[float] = []
    channel_values: list[list[float]] = []
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            metadata_line = handle.readline()
            header_line = handle.readline()
            metadata = parse_metadata_line(metadata_line)
            header = next(csv.reader([header_line])) if header_line else []
            row.update(metadata)
            channel_names = [
                name for name in header
                if name.startswith("EEG.") and name not in {
                    "EEG.Counter", "EEG.Interpolated", "EEG.RawCq", "EEG.Battery",
                    "EEG.BatteryPercent", "EEG.MarkerHardware",
                }
            ]
            row["channel_names"] = ";".join(channel_names)
            row["channel_count"] = len(channel_names)
            index = {name: position for position, name in enumerate(header)}
            timestamp_index = index.get("Timestamp")
            counter_index = index.get("EEG.Counter")
            interpolated_index = index.get("EEG.Interpolated")
            channel_indices = [index[name] for name in channel_names]
            channel_values = [[] for _ in channel_names]
            row_count = 0
            malformed_rows = 0
            nonfinite_cells = 0
            missing_sample_rows = 0
            interpolated_rows = 0
            for values in csv.reader(handle):
                if not values or all(not norm(value) for value in values):
                    continue
                row_count += 1
                if timestamp_index is None or len(values) <= timestamp_index:
                    timestamp = None
                else:
                    timestamp = finite_float(values[timestamp_index])
                if timestamp is not None:
                    timestamps.append(timestamp)
                if counter_index is not None and len(values) > counter_index:
                    counter = finite_float(values[counter_index])
                    if counter is not None:
                        counters.append(counter)
                if interpolated_index is not None and len(values) > interpolated_index:
                    if norm(values[interpolated_index]).lower() in {"1", "1.0", "true"}:
                        interpolated_rows += 1
                finite_in_row = 0
                for channel_position, source_index in enumerate(channel_indices):
                    value = finite_float(values[source_index]) if len(values) > source_index else None
                    if value is None:
                        nonfinite_cells += 1
                    else:
                        finite_in_row += 1
                        channel_values[channel_position].append(value)
                if len(values) < (max(channel_indices, default=-1) + 1):
                    malformed_rows += 1
                if timestamp is None or finite_in_row == 0:
                    missing_sample_rows += 1
            row["file_readable"] = "true"
            row["row_count"] = row_count
            row["malformed_row_count"] = malformed_rows
            row["timestamp_count"] = len(timestamps)
            row["counter_count"] = len(counters)
            row["interpolated_row_count"] = interpolated_rows
            row["missing_sample_rows"] = missing_sample_rows
            row["missing_channel_cells"] = nonfinite_cells
            total_channel_cells = row_count * len(channel_names)
            row["expected_channel_cells"] = total_channel_cells
            row["nonfinite_proportion"] = csv_float(nonfinite_cells / total_channel_cells if total_channel_cells else 1.0)
            row["device_model"] = {
                "EPOCX": "Epoc X",
                "EPOCPLUS": "Epoc+",
            }.get(norm(metadata.get("headset_type")).upper(), norm(metadata.get("headset_type")))
            sampling_match = re.search(r"eeg_(\d+(?:\.\d+)?)", norm(metadata.get("sampling_rate_metadata")), re.I)
            row["sampling_rate_hz"] = csv_float(float(sampling_match.group(1)) if sampling_match else None)
            row["duration_seconds"] = csv_float(
                float(metadata["stop_timestamp"] - metadata["start_timestamp"])
                if isinstance(metadata.get("start_timestamp"), (int, float))
                and isinstance(metadata.get("stop_timestamp"), (int, float)) else None
            )
            finite_timestamps = timestamps
            deltas = [b - a for a, b in zip(finite_timestamps, finite_timestamps[1:])]
            duplicate_timestamps = sum(1 for delta in deltas if delta == 0)
            nonmonotonic_timestamps = sum(1 for delta in deltas if delta < 0)
            sampling_rate = float(row["sampling_rate_hz"]) if row.get("sampling_rate_hz") not in {"", None} else None
            expected_delta = 1.0 / sampling_rate if sampling_rate else None
            gap_limit = max(0.25, 2.5 * expected_delta) if expected_delta else None
            timestamp_gaps = [delta for delta in deltas if gap_limit is not None and delta > gap_limit]
            row["duplicate_timestamp_count"] = duplicate_timestamps
            row["nonmonotonic_timestamp_count"] = nonmonotonic_timestamps
            row["timestamp_gap_count"] = len(timestamp_gaps)
            row["longest_timestamp_gap_seconds"] = csv_float(max(timestamp_gaps) if timestamp_gaps else 0.0)
            counter_deltas = [b - a for a, b in zip(counters, counters[1:])]
            row["counter_duplicate_or_nonmonotonic_count"] = sum(1 for delta in counter_deltas if delta <= 0)
            finite_per_channel = [len(values) for values in channel_values]
            row["flat_channel_count"] = sum(
                1 for values in channel_values if values and max(values) == min(values)
            )
            row["flat_channel_names"] = ";".join(
                channel_names[position] for position, values in enumerate(channel_values)
                if values and max(values) == min(values)
            )
            row["gross_amplitude_outlier_proportion"] = csv_float(robust_outlier_fraction(channel_values))
            row["gross_clipping_saturation_proportion"] = csv_float(saturation_fraction(channel_values))
            row["recording_interruptions"] = len(timestamp_gaps)
            warnings: list[str] = []
            if row["channel_count"] != 14:
                warnings.append("unexpected_14_channel_count")
            if row["malformed_row_count"]:
                warnings.append("malformed_rows")
            if row["duplicate_timestamp_count"] or row["nonmonotonic_timestamp_count"]:
                warnings.append("timestamp_order_problem")
            if row["flat_channel_count"]:
                warnings.append("flat_channel")
            row["structural_warning"] = ";".join(warnings)
            row["structurally_usable"] = bool_text(
                row["file_readable"] == "true"
                and row["row_count"] > 0
                and row["channel_count"] == 14
                and row["timestamp_count"] == row["row_count"]
                and not row["malformed_row_count"]
            )
    except Exception as exc:  # pragma: no cover - defensive file audit path
        row["structural_warning"] = f"parse_error:{type(exc).__name__}:{exc}"
        row["structurally_usable"] = "false"
    return row


def parse_eye(path: Path, candidate: bool) -> dict[str, object]:
    row: dict[str, object] = {
        "source_file": str(path.relative_to(ROOT)),
        "candidate_complete_modalities": bool_text(candidate),
        "file_readable": "false",
    }
    try:
        header, values_rows, readable, error = read_csv_rows(path)
        row["file_readable"] = bool_text(readable)
        if not readable:
            row["structural_warning"] = error
            row["structurally_usable"] = "false"
            return row
        row["field_names"] = ";".join(header)
        index = {name: position for position, name in enumerate(header)}
        required = ["timeChecked", "weight", "looked_col", "looked_row"]
        missing_fields = [field for field in required if field not in index]
        row["required_fields_present"] = bool_text(not missing_fields)
        row["missing_required_fields"] = ";".join(missing_fields)
        times: list[float] = []
        valid_times: list[float] = []
        rows = 0
        malformed_rows = 0
        valid_rows = 0
        invalid_time_rows = 0
        invalid_gaze_rows = 0
        implausible_rows = 0
        columns: list[float] = []
        rows_coordinate: list[float] = []
        time_index = index.get("timeChecked")
        col_index = index.get("looked_col")
        row_index = index.get("looked_row")
        for values in values_rows:
            if not values or all(not norm(value) for value in values):
                continue
            rows += 1
            if len(values) < len(header):
                malformed_rows += 1
            timestamp = finite_float(values[time_index]) if time_index is not None and len(values) > time_index else None
            looked_col = finite_float(values[col_index]) if col_index is not None and len(values) > col_index else None
            looked_row = finite_float(values[row_index]) if row_index is not None and len(values) > row_index else None
            if timestamp is None:
                invalid_time_rows += 1
            else:
                times.append(timestamp)
            valid = timestamp is not None and looked_col is not None and looked_row is not None
            if not valid:
                invalid_gaze_rows += 1
            else:
                valid_rows += 1
                valid_times.append(timestamp)
                columns.append(looked_col)
                rows_coordinate.append(looked_row)
                # The documented Robots grid is 47 columns by 14 rows and
                # the local files use one-based integer positions.
                if looked_col < 1 or looked_col > 47 or looked_row < 1 or looked_row > 14:
                    implausible_rows += 1
        deltas = [b - a for a, b in zip(times, times[1:])]
        positive_deltas = [delta for delta in deltas if delta > 0]
        median_dt = statistics.median(positive_deltas) if positive_deltas else None
        valid_deltas = [b - a for a, b in zip(valid_times, valid_times[1:])]
        duplicate = sum(1 for delta in deltas if delta == 0)
        nonmonotonic = sum(1 for delta in deltas if delta < 0)
        discontinuity_limit = max(0.1, 2.5 * median_dt) if median_dt else None
        discontinuities = [delta for delta in valid_deltas if discontinuity_limit is not None and delta > discontinuity_limit]
        longest_gap = max((delta - median_dt for delta in discontinuities), default=0.0) if median_dt else None
        duration = max(times) - min(times) if times else None
        valid_duration = max(valid_times) - min(valid_times) if valid_times else None
        expected_nominal_samples = int(round(duration / median_dt)) + 1 if duration is not None and median_dt else None
        row.update({
            "row_count": rows,
            "malformed_row_count": malformed_rows,
            "valid_gaze_rows": valid_rows,
            "invalid_time_rows": invalid_time_rows,
            "invalid_gaze_rows": invalid_gaze_rows,
            "valid_gaze_coverage": csv_float(valid_rows / rows if rows else 0.0),
            "missing_gaze_proportion": csv_float(1.0 - valid_rows / rows if rows else 1.0),
            "duration_seconds": csv_float(duration),
            "valid_time_span_seconds": csv_float(valid_duration),
            "relative_start_seconds": csv_float(min(times) if times else None),
            "relative_end_seconds": csv_float(max(times) if times else None),
            "valid_relative_start_seconds": csv_float(min(valid_times) if valid_times else None),
            "valid_relative_end_seconds": csv_float(max(valid_times) if valid_times else None),
            "median_sample_interval_seconds": csv_float(median_dt),
            "nominal_sampling_hz": csv_float(1.0 / median_dt if median_dt else None),
            "expected_nominal_samples": expected_nominal_samples if expected_nominal_samples is not None else "",
            "observed_to_expected_sample_ratio": csv_float(valid_rows / expected_nominal_samples if expected_nominal_samples else None),
            "longest_missing_gap_seconds": csv_float(longest_gap),
            "discontinuity_count": len(discontinuities),
            "duplicate_timestamp_count": duplicate,
            "nonmonotonic_timestamp_count": nonmonotonic,
            "left_eye_fields_present": bool_text(any("left" in name.lower() for name in header)),
            "right_eye_fields_present": bool_text(any("right" in name.lower() for name in header)),
            "both_eye_fields_present": bool_text(any("left" in name.lower() for name in header) and any("right" in name.lower() for name in header)),
            "looked_col_min": csv_float(min(columns) if columns else None),
            "looked_col_max": csv_float(max(columns) if columns else None),
            "looked_row_min": csv_float(min(rows_coordinate) if rows_coordinate else None),
            "looked_row_max": csv_float(max(rows_coordinate) if rows_coordinate else None),
            "implausible_coordinate_rows": implausible_rows,
            "task_coverage_class": (
                "no_valid_gaze" if not valid_times else
                "throughout_task_anchor" if valid_times[0] <= 1.0 and valid_times[-1] >= 0.9 * (duration or 0.0) else
                "intermittent_or_partial"
            ),
            "structural_warning": ";".join(
                warning for warning in [
                    "missing_required_fields" if missing_fields else "",
                    "malformed_rows" if malformed_rows else "",
                    "duplicate_timestamps" if duplicate else "",
                    "nonmonotonic_timestamps" if nonmonotonic else "",
                    "implausible_coordinates" if implausible_rows else "",
                ] if warning
            ),
            "structurally_usable": bool_text(bool(readable and not missing_fields and rows > 0 and valid_rows > 0 and not malformed_rows)),
        })
    except Exception as exc:  # pragma: no cover - defensive file audit path
        row["structural_warning"] = f"parse_error:{type(exc).__name__}:{exc}"
        row["structurally_usable"] = "false"
    return row


def parse_behavior(path: Path, candidate: bool) -> dict[str, object]:
    row: dict[str, object] = {
        "source_file": str(path.relative_to(ROOT)),
        "candidate_complete_modalities": bool_text(candidate),
        "file_readable": "false",
    }
    try:
        header, values_rows, readable, error = read_csv_rows(path)
        row["file_readable"] = bool_text(readable)
        if not readable:
            row["structural_warning"] = error
            row["structurally_usable"] = "false"
            return row
        lowered = [name.lower() for name in header]
        response_fields = [name for name in header if any(token in name.lower() for token in ["omision", "comision", "aciertos", "velocidadtrabajo"])]
        success_error_fields = [name for name in header if any(token in name.lower() for token in ["aciertos", "omision", "comision", "correct", "error", "success"])]
        target_fields = [name for name in header if any(token in name.lower() for token in ["target", "object", "robot", "bot"])]
        timestamp_fields = [name for name in header if any(token in name.lower() for token in ["timestamp", "time", "date"])]
        has_summary_schema = bool(response_fields) and any(name.lower() == "maxbots" for name in header)
        row_count = len([values for values in values_rows if values and any(norm(value) for value in values)])
        malformed = sum(1 for values in values_rows if len(values) < len(header))
        row.update({
            "field_names": ";".join(header),
            "record_count": row_count,
            "number_task_events": 0,
            "event_types": "session_summary" if row_count else "",
            "response_information_present": bool_text(bool(response_fields)),
            "success_error_information_present": bool_text(bool(success_error_fields)),
            "target_object_identifier_fields": ";".join(target_fields),
            "target_object_identifiers_present": "false",
            "timestamp_fields": ";".join(timestamp_fields),
            "timestamp_completeness": "no_timestamp_field" if not timestamp_fields else "not_assessed",
            "duplicate_timestamps": "not_applicable" if not timestamp_fields else "not_assessed",
            "nonmonotonic_timestamps": "not_applicable" if not timestamp_fields else "not_assessed",
            "missing_intervals": "not_assessable_without_event_timestamps",
            "duration_seconds": "",
            "task_progression_reconstructable": "limited_session_summary_only" if has_summary_schema else "no",
            "structural_warning": ";".join(warning for warning in ["malformed_rows" if malformed else "", "no_summary_schema" if not has_summary_schema else ""] if warning),
            "structurally_usable": bool_text(bool(row_count and not malformed and has_summary_schema)),
        })
    except Exception as exc:  # pragma: no cover - defensive file audit path
        row["structural_warning"] = f"parse_error:{type(exc).__name__}:{exc}"
        row["structurally_usable"] = "false"
    return row










