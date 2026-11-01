"""Recompute primary, ladder and diagnosis models using immutable feature inputs."""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import sys
sys.dont_write_bytecode = True
import importlib.util
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from replay_status import classify_replay

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ["EEG_RELEASE_ROOT"])
SOURCE = ROOT / "audit/preregistered_analysis"
OUT = Path(os.environ["EEG_RUN_ROOT"]) / "models"
RAW = Path(os.environ["EEG_DATASET_ROOT"])
spec = importlib.util.spec_from_file_location("g1_primary", SOURCE / "run_preregistered_analysis.py")
primary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(primary)

def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

def metrics(frame, column):
    rows = []
    for repeat, group in frame.groupby("repeat"):
        y, p = group.y.to_numpy(int), group[column].to_numpy(float)
        pc = np.clip(p, 1e-8, 1 - 1e-8)
        # Independently implemented pairwise concordance, including exact ties.
        contrasts = p[y == 1, None] - p[None, y == 0]
        rows.append(dict(repeat=int(repeat), logloss=float(np.mean(-y*np.log(pc)-(1-y)*np.log1p(-pc))),
                         brier=float(np.mean((y-p)**2)), auroc=float(np.mean((contrasts > 0) + .5*(contrasts == 0)))))
    return pd.DataFrame(rows)

def check_folds(folds, path):
    accepted = pd.read_csv(path)
    ids = np.asarray(folds["ids"])
    count = 0
    for outer in folds["outer"]:
        sub = accepted[(accepted.repeat == outer["repeat"]) & (accepted.outer_fold == outer["outer_fold"])]
        tr, te = set(outer["outer_train"]), set(outer["outer_test"])
        assert not tr & te and tr | te == set(range(len(ids)))
        for role, indices in (("train", tr), ("test", te)):
            assert set(sub[(sub.level == "outer") & (sub.role == role)].participant_id) == set(ids[list(indices)])
            count += 1
        seen = []
        for inner in outer["inner_folds"]:
            a, b = set(inner["inner_train"]), set(inner["inner_valid"])
            assert not a & b and a | b == tr and not te & (a | b)
            seen.extend(b)
            ss = sub[(sub.level == "inner") & (sub.inner_fold == inner["inner_fold"])]
            for role, indices in (("train", a), ("valid", b)):
                actual_role = "validation" if role == "valid" and "validation" in set(ss.role) else role
                assert set(ss[ss.role == actual_role].participant_id) == set(ids[list(indices)])
                count += 1
        assert sorted(seen) == sorted(tr)
    return count

DATA = None
def initialize(data):
    global DATA
    DATA = data
    threadpool_limits(1)

def worker(job):
    outcome, mode, family, index = job
    raw, full = DATA[outcome]
    folds = dict(full, outer=[full["outer"][index]])
    failures = []
    result = primary.run_tuned_model(folds, raw, mode, family, failures)
    return outcome, family, result, failures

def main():
    OUT.mkdir(exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    primary.METADATA_PATH = RAW / "users_demographics.json"
    base, measurement = primary.load_measurements()  # Released measurement and estimator identity checks remain enabled.
    linked, linkage = primary.link_outcomes(base[["participant_id"]])
    records = json.loads(primary.METADATA_PATH.read_text(encoding="utf-8"))
    by_id = {r["user"]: r for r in records}
    assert len(by_id) == len(records)
    ids = measurement["ids"]
    y = np.array([1 if str(by_id[pid]["group"]) == "1" else 0 for pid in ids])
    assert all(str(by_id[pid]["group"]) in ("1", "2") for pid in ids)
    assert np.array_equal(y, linked.group_y.to_numpy(int))
    saved = pd.read_csv(SOURCE / "PRIMARY_OUTER_PREDICTIONS.csv")
    assert len(saved) == 960 and not saved.duplicated(["repeat", "participant_id"]).any()
    for _, row in saved.iterrows():
        assert row.y == y[ids.index(row.participant_id)]
    diag_indices = [i for i, pid in enumerate(ids) if str(by_id[pid].get("diagnosed", "")).strip().lower() in ("yes", "no")]
    diag_ids = [ids[i] for i in diag_indices]
    dy = np.array([str(by_id[pid]["diagnosed"]).strip().lower() == "yes" for pid in diag_ids], int)
    group_folds = primary.build_folds(ids, y, "group")
    diag_folds = primary.build_folds(diag_ids, dy, "diagnosed")
    checks = {"group": check_folds(group_folds, SOURCE / "PRIMARY_FOLD_MANIFEST.csv"),
              "diagnosed": check_folds(diag_folds, SOURCE / "DIAGNOSED_FOLD_MANIFEST.csv")}
    data = {"group": (primary.prepare_raw(base, measurement), group_folds),
            "diagnosed": (primary.prepare_raw(base, measurement, diag_indices), diag_folds)}
    dump(OUT / "preflight.json", dict(started_utc=started, input_hash_guards=len(primary.expected_artifact_hashes()),
         group_n=len(ids), group_positive=int(y.sum()), diagnosed_n=len(dy), diagnosed_positive=int(dy.sum()),
         exact_fold_membership_checks=checks, raw_metadata_labels_equal=True,
         path_binding_changes={"METADATA_PATH": str(primary.METADATA_PATH)},
         scientific_functions_modified=False, accepted_output_directories_written=False))
    modes = [("group", mode, "adaptive_M3" if mode == "adaptive" else mode) for mode in ("M0", "M1", "M2", "adaptive", "E0", "E1", "E2")]
    modes += [("diagnosed", "M2", "M2_diagnosed"), ("diagnosed", "adaptive", "adaptive_M3_diagnosed")]
    jobs = [(outcome, mode, family, i) for outcome, mode, family in modes for i in range(50)]
    predictions, selection, failures = [], [], []
    with ProcessPoolExecutor(max_workers=int(os.environ["EEG_JOBS"]), initializer=initialize, initargs=(data,)) as pool:
        futures = [pool.submit(worker, job) for job in jobs]
        for count, future in enumerate(as_completed(futures), 1):
            _, _, result, failed = future.result()
            predictions.extend(result["predictions"])
            selection.extend(result["selection"])
            failures.extend(failed)
            if count % 25 == 0:
                print(f"Nested model replay: {count}/{len(jobs)} outer fits", flush=True)
    p = pd.DataFrame(predictions).sort_values(["outcome", "model_family", "repeat", "participant_index"])
    s = pd.DataFrame(selection).sort_values(["outcome", "model_family", "repeat", "outer_fold", "representation", "C"])
    p.to_csv(OUT / "predictions.csv", index=False, float_format="%.17g")
    s.to_csv(OUT / "selection.csv", index=False, float_format="%.17g")
    dump(OUT / "failures.json", failures)
    checks_out = []
    fields = {"M0": ("p_m0", "m0"), "M1": ("p_m1", "m1"), "M2": ("p_m2", "m2"),
              "adaptive_M3": ("p_adaptive_m3", "m3"), "E0": ("p_e0", "e0"), "E1": ("p_e1", "e1"), "E2": ("p_e2", "e2")}
    for family, (column, prefix) in fields.items():
        merged = p[p.model_family == family].merge(saved, on=["repeat", "outer_fold", "participant_id", "y"], validate="one_to_one")
        assert len(merged) == 960
        error = float(np.max(np.abs(merged.probability - merged[column])))
        choices = bool((merged.C == merged[prefix + "_C"]).all() and (merged.representation == merged[prefix + "_representation"]).all())
        checks_out.append(dict(family=family, rows=len(merged), max_probability_error=error, choices_exact=choices, passed=error <= 1e-7 and choices))
    canonical = pd.read_csv(SOURCE / "PRIMARY_INNER_SELECTION.csv")
    keys = ["outcome", "model_family", "repeat", "outer_fold", "representation", "C"]
    m = s[s.outcome == "group"].merge(canonical, on=keys, suffixes=("_new", "_old"), validate="one_to_one")
    assert len(m) == len(canonical)
    inner_error = float(np.max(np.abs(m.pooled_logloss_new - m.pooled_logloss_old)))
    d2 = metrics(p[p.model_family == "M2_diagnosed"], "probability").set_index("repeat")
    d3 = metrics(p[p.model_family == "adaptive_M3_diagnosed"], "probability").set_index("repeat")
    reference = pd.read_csv(SOURCE / "DIAGNOSED_SENSITIVITY.csv").set_index("repeat")
    derr = {}
    for metric in ("logloss", "brier", "auroc"):
        for model, frame in (("m2", d2), ("m3", d3)):
            derr[model + "_" + metric] = float(np.max(np.abs(frame[metric] - reference[model + "_" + metric])))
    status, mismatches = classify_replay(checks_out, m, derr, failures,
                                        strict=os.environ.get('EEG_STRICT_INTERMEDIATES') == '1')
    mismatches.to_csv(OUT / 'INTERMEDIATE_SCORE_DIFFERENCES.csv', index=False, float_format='%.17g')
    result = dict(started_utc=started, completed_utc=datetime.now(timezone.utc).isoformat(),
                  original_model_checks=checks_out, primary_inner_score_max_error=inner_error,
                  primary_inner_records=len(m), diagnosed_repeat_metric_max_errors=derr,
                  failed_model_operations=len(failures), outer_fits=len(jobs),
                  **status)
    dump(OUT / "REPLAY_CHECKS.json", result)
    print(json.dumps(result, indent=2), flush=True)
    if not result['command_completed']:
        raise RuntimeError('Reproduction checks failed; inspect REPLAY_CHECKS.json')
    if not result['intermediate_scores_passed']:
        print('WARNING: predictive results and selections reproduced; intermediate scores differ. '
              'The 1e-7 numerical check has NOT passed. See INTERMEDIATE_SCORE_DIFFERENCES.csv.', flush=True)

if __name__ == "__main__":
    main()
