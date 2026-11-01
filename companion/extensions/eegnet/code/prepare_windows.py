"""Reconstruct frozen accepted windows without modifying the reference release."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import numpy as np
import pandas as pd
sys.dont_write_bytecode = True

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()

def write(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

def cache_fingerprint(manifest):
    stable=dict(reference_inputs=manifest['reference_inputs'],participants=[{k:item[k] for k in ('participant_id','raw_file','raw_sha256','cache_file','cache_sha256','shape')} for item in manifest['participants']])
    return hashlib.sha256(json.dumps(stable,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def one(args):
    release,raw,out,record,expected,expected_qc=args
    release,raw,out=map(Path,(release,raw,out))
    s=load(release/'audit/stage1_blind_feature_extraction/run_stage1_extraction.py','stage')
    b=load(release/'audit/exploratory_eeg_battery/exploratory_eeg_battery.py','battery')
    pid=record['participant_id']
    rel=Path(*record['eeg_source_file'].replace('\\','/').split('/')[1:])
    path=(raw/rel).resolve()
    if not path.is_relative_to(raw.resolve()):raise ValueError('Raw path leaves root')
    got=b.stage1_capture(s,dict(participant_id=pid,session_id=str(record['session_id']),eeg_path=path),{pid:{'age':expected['age']}})
    qc=got['qc'];assert qc['preprocessing_status']=='PASS'
    keys=['candidate_window_count','rejected_window_count','clean_window_fraction','clean_seconds','bad_channel_count']
    for k in keys: assert abs(float(qc[k])-float(expected_qc[k]))<1e-8,(pid,k)
    err=max(abs(float(got['e0'][k])-float(expected[k])) for k in s.EEG_FEATURES)
    assert err<1e-8,(pid,err)
    windows=np.stack(got['accepted_windows']).transpose(0,2,1).astype(np.float32)
    assert windows.shape[1:]==(14,256) and np.isfinite(windows).all()
    target=out/(pid+'.npy')
    if target.exists():raise FileExistsError(target)
    np.save(target,windows,allow_pickle=False)
    return dict(participant_id=pid,raw_file=rel.as_posix(),raw_sha256=sha(path),cache_file=target.name,
                cache_sha256=sha(target),shape=list(windows.shape),e0_max_abs_error=err,
                qc={k:qc[k] for k in keys})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--release',type=Path,required=True);ap.add_argument('--raw',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
    if a.out.exists():raise FileExistsError('Output directory must be new')
    a.out.mkdir(parents=True)
    s=a.release/'audit/stage1_blind_feature_extraction'
    base=pd.read_csv(s/'BLIND_PRIMARY_FEATURE_MATRIX.csv').set_index('participant_id')
    qc=pd.read_csv(s/'EEG_PREPROCESSING_QC.csv').set_index('participant_id')
    roster=pd.read_csv(s/'BLIND_COHORT_MANIFEST.csv',dtype={'session_id':str}).set_index('participant_id').loc[base.index].reset_index()
    assert len(base)==96 and len(roster)==96
    files=['audit/stage1_blind_feature_extraction/run_stage1_extraction.py','audit/exploratory_eeg_battery/exploratory_eeg_battery.py','audit/stage1_blind_feature_extraction/BLIND_PRIMARY_FEATURE_MATRIX.csv','audit/stage1_blind_feature_extraction/BLIND_COHORT_MANIFEST.csv','audit/stage1_blind_feature_extraction/EEG_PREPROCESSING_QC.csv','audit/preregistered_analysis/PRIMARY_FOLD_MANIFEST.csv','audit/preregistered_analysis/PRIMARY_OUTER_PREDICTIONS.csv','audit/preregistered_analysis/OUTCOME_LINKAGE_MANIFEST.csv']
    inputs={f:sha(a.release/f) for f in files}
    items=[]
    with ProcessPoolExecutor(a.workers) as ex:
        jobs=[ex.submit(one,(str(a.release),str(a.raw),str(a.out),r,base.loc[r['participant_id']].to_dict(),qc.loc[r['participant_id']].to_dict())) for r in roster.to_dict('records')]
        for f in as_completed(jobs):
            item=f.result();items.append(item);print('verified',len(items),'/96',item['participant_id'],flush=True)
    write(a.out/'INPUT_MANIFEST.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),reference_inputs=inputs,participants=sorted(items,key=lambda x:x['participant_id']),cache_dtype='float32',axis_order=['window','channel','sample'],source='Frozen accepted windows, regenerated and checked; no cohort changes'))

if __name__=='__main__':main()
