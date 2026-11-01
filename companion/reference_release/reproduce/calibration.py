"""Recalculate scalar calibration and reliability bins from saved outer predictions."""
import importlib.util, json, os
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(os.environ["EEG_RELEASE_ROOT"])
OUT=Path(os.environ["EEG_RUN_ROOT"])/"calibration"

def main():
    spec=importlib.util.spec_from_file_location("primary",ROOT/"audit/preregistered_analysis/run_preregistered_analysis.py")
    primary=importlib.util.module_from_spec(spec);spec.loader.exec_module(primary)
    pred=pd.read_csv(ROOT/"audit/preregistered_analysis/PRIMARY_OUTER_PREDICTIONS.csv").sort_values(["repeat","participant_index"])
    rows=[];failures=[]
    for model,column in [("M2","p_m2"),("adaptive_M3","p_adaptive_m3")]:
        for repeat in sorted(pred.repeat.unique()):
            sub=pred[pred.repeat.eq(repeat)]
            rows.extend(primary.calibration_rows(sub.y.to_numpy(dtype=int),sub[column].to_numpy(dtype=np.float64),sub.participant_id.tolist(),int(repeat),model,failures,"group"))
    OUT.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows, columns=['outcome', 'model', 'repeat', 'record_type', 'intercept', 'slope', 'citl', 'bin', 'n', 'mean_probability', 'observed_rate', 'failure_code']).to_csv(OUT/"CALIBRATION_RESULTS.csv",index=False)
    (OUT/"CALIBRATION_FAILURES.json").write_text(json.dumps(failures,indent=2)+"\n",encoding="utf-8")
    print("Calibration recalculated from archived outer predictions; inspect scalar failure flags and compare reliability bins separately.")
if __name__=="__main__":main()
