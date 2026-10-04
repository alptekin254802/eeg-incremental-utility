"""Re-extract all accepted EEG families and non-EEG predictors without altering originals."""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import sys
sys.dont_write_bytecode = True
import importlib.util
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ["EEG_RELEASE_ROOT"])
AUDIT = ROOT / "audit"
RAW = Path(os.environ["EEG_DATASET_ROOT"])
OUT = Path(os.environ["EEG_RUN_ROOT"]) / "raw_features"

def module(rel, name):
    spec = importlib.util.spec_from_file_location(name, AUDIT / rel)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj
STAGE = module("stage1_blind_feature_extraction/run_stage1_extraction.py", "g1_stage")
ENRICH = module("stage1_5_enriched_eeg_feasibility/run_stage1_5_audit.py", "g1_enrich")
BATTERY = module("exploratory_eeg_battery/exploratory_eeg_battery.py", "g1_battery")

def rawpath(text):
    path = RAW.joinpath(*text.replace(chr(92),"/").split("/")[1:]).resolve()
    assert path.is_relative_to(RAW.resolve()) and path.is_file()
    return path

def compare(observed, expected):
    a, b = np.asarray(observed, float).ravel(), np.asarray(expected, float).ravel()
    assert a.shape == b.shape
    return float(np.max(np.abs(a-b))), bool(np.allclose(a, b, atol=1e-10, rtol=1e-8))

def worker(payload):
    threadpool_limits(1)
    pid, roster, qc, expected, blocks, cov = payload
    path = rawpath(roster["eeg_source_file"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    captured = BATTERY.stage1_capture(STAGE, dict(participant_id=pid, session_id=roster["session_id"], eeg_path=path), {pid: {"age": 1}})
    # Age is only checked for presence by this EEG-only function, never used in EEG computations.
    result = dict(participant_id=pid, raw_sha256=digest, raw_relative_path=str(path.relative_to(RAW)),
                  status=captured["qc"]["preprocessing_status"], status_exact=captured["qc"]["preprocessing_status"] == qc["preprocessing_status"],
                  bad_channel_count_exact=captured["qc"]["bad_channel_count"] == qc["bad_channel_count"])
    assert result["status_exact"] and result["bad_channel_count_exact"]
    if result["status"] != "PASS":
        result["all_checks_passed"] = True
        (OUT / (pid + ".json")).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    qc_errors = {}
    for key in ("candidate_window_count", "rejected_window_count", "clean_window_fraction", "clean_seconds"):
        difference = abs(float(captured["qc"][key]) - float(qc[key]))
        qc_errors[key] = difference
        if key.endswith("_count"):
            assert difference == 0, (pid, key, difference)
        else:
            assert difference <= 1e-12, (pid, key, difference)
    result["qc_max_absolute_error"] = max(qc_errors.values())
    windows, segments, indices = captured["accepted_windows"], captured["segments"], captured["accepted_ids"]
    result["retained_windows"] = len(windows)
    result["qc_counts_exact"] = True
    e0 = captured["e0"]
    behavior = STAGE.extract_behavior(rawpath(roster["behavior_source_file"]))
    gaze = STAGE.extract_gaze(rawpath(roster["eye_source_file"]))
    values = {**e0, **behavior, **gaze}
    error, ok = compare(list(values.values()), [expected[n] for n in values])
    result.update(e0_behavior_gaze_max_error=error, e0_behavior_gaze_passed=ok)
    topo, temporal, entropy, topo_names, temp_names, entropy_names = ENRICH.e1_features(windows)
    for name, values, names in (("topographic", topo, topo_names), ("temporal", temporal, temp_names), ("spectral_entropy", entropy, entropy_names)):
        error, ok = compare(values, [blocks[name][n] for n in names])
        result[name + "_max_error"], result[name + "_passed"] = error, ok
    matrices, _ = ENRICH.e2_covariance(segments, indices, ENRICH.helmert_contrast_basis(14))
    result["e2_max_error"], result["e2_passed"] = compare(matrices, cov)
    for family, values in (("E3", BATTERY.e3_extract(windows)), ("E4", BATTERY.e4_extract(windows)), ("E5", BATTERY.e5_extract(segments, indices))):
        result[family + "_max_error"], result[family + "_passed"] = compare(list(values.values()), [blocks[family][n] for n in values])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    result["all_checks_passed"] = all(v for k, v in result.items() if k.endswith("_passed"))
    (OUT / (pid + ".json")).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result

def main():
    OUT.mkdir(exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    stage_dir = AUDIT / "stage1_blind_feature_extraction"
    roster = pd.read_csv(stage_dir / "BLIND_COHORT_MANIFEST.csv").set_index("participant_id")
    qc = pd.read_csv(stage_dir / "EEG_PREPROCESSING_QC.csv").set_index("participant_id")
    features = pd.read_csv(stage_dir / "BLIND_PRIMARY_FEATURE_MATRIX.csv").set_index("participant_id")
    blocks = {}
    for name, path in [("topographic", "E1_TOPOGRAPHIC_SPECTRAL_RAW.csv"), ("temporal", "E1_TEMPORAL_IQR_RAW.csv"), ("spectral_entropy", "E1_REGIONAL_ENTROPY.csv")]:
        blocks[name] = pd.read_csv(AUDIT / "stage1_5_enriched_eeg_feasibility" / path).set_index("participant_id")
    for family in ("E3", "E4", "E5"):
        blocks[family] = pd.read_csv(AUDIT / "exploratory_eeg_battery" / (family + "_RAW_FEATURES.csv")).set_index("participant_id")
    with np.load(AUDIT / "stage1_5_enriched_eeg_feasibility/E2_PARTICIPANT_COVARIANCES.npz", allow_pickle=False) as archive:
        covariance = dict(zip(archive["participant_id"].tolist(), archive["covariances"]))
    payloads = [(pid, roster.loc[pid].to_dict(), qc.loc[pid].to_dict(), features.loc[pid].to_dict() if pid in features.index else {},
                 {k: b.loc[pid].to_dict() for k,b in blocks.items()} if pid in features.index else {}, covariance.get(pid)) for pid in roster.index]
    results = []
    with ProcessPoolExecutor(max_workers=int(os.environ["EEG_JOBS"])) as pool:
        futures = [pool.submit(worker, payload) for payload in payloads]
        for i, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if i % 10 == 0 or i == len(payloads):
                print(f"Raw EEG and feature replay: {i}/{len(payloads)} participants", flush=True)
    frame = pd.DataFrame(results).sort_values("participant_id")
    frame.to_csv(OUT / "RAW_REPLAY_CHECKS.csv", index=False, float_format="%.17g")
    summary = dict(started_utc=started, completed_utc=datetime.now(timezone.utc).isoformat(), participant_count=len(frame),
                   preprocessing_status_counts=frame.status.value_counts().to_dict(), passed=bool(frame.all_checks_passed.all()),
                   maximum_errors={k: float(frame[k].max()) for k in frame if k.endswith("_max_error")},
                   scope="Starts at the saved 98-person measurement-eligible roster. Earlier 126-to-98 screen not rerun. All 96 accepted E0-E5 raw features plus behavior/gaze re-extracted. No model results changed.")
    (OUT / "RAW_REPLAY_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    assert summary["passed"]

if __name__ == "__main__":
    main()
