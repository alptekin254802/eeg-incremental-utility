"""Post-hoc window-wise Fourier-phase controls, with raw reconstruction."""
from __future__ import annotations
import os
for _key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[_key]="1"
import sys
sys.dont_write_bytecode=True
import argparse
import hashlib
import itertools
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import run_posthoc as core

HERE=core.HERE
OUT=HERE/"signal_controls"
RAW=Path(os.environ["EEG_DATASET_ROOT"])
STAGE=core.load_module(core.AUDIT/"stage1_blind_feature_extraction/run_stage1_extraction.py","frozen_stage1")
BATTERY=core.load_module(core.AUDIT/"exploratory_eeg_battery/exploratory_eeg_battery.py","frozen_battery")
REGIONS=[[BATTERY.CHANNELS.index(c) for c in v] for v in BATTERY.REGIONS.values()]
NAMES=[f"{r}_{q}" for r in BATTERY.REGIONS for q in ("perm_entropy","hjorth_mobility","hjorth_complexity")]
LUT=np.full(3125,-1,dtype=int)
for i,p in enumerate(itertools.permutations(range(5))):LUT[np.ravel_multi_index(p,(5,)*5)]=i

def e4_vectorized(windows):
    # (accepted-window, channel, sample). Same normalization and finite differences as the frozen code.
    x=np.asarray(windows,dtype=float).transpose(0,2,1)
    med=np.median(x,axis=2,keepdims=True)
    scale=1.4826*np.median(np.abs(x-med),axis=2,keepdims=True)
    assert np.isfinite(x).all() and (scale>0).all()
    x=(x-med)/scale
    v=np.var(x,axis=2);v1=np.var(np.diff(x,axis=2),axis=2);v2=np.var(np.diff(x,n=2,axis=2),axis=2)
    assert (v>0).all() and (v1>0).all() and (v2>0).all()
    mobility=np.sqrt(v1/v);complexity=np.sqrt(v2/v1)/mobility
    orders=np.argsort(np.lib.stride_tricks.sliding_window_view(x,5,axis=2),axis=-1,kind="stable")
    codes=(orders*np.array([625,125,25,5,1])).sum(axis=-1)
    ranks=LUT[codes].reshape(-1,252)
    assert (ranks>=0).all()
    offsets=np.arange(len(ranks))[:,None]*120
    counts=np.bincount((ranks+offsets).ravel(),minlength=len(ranks)*120).reshape(-1,120)
    probability=counts/252.
    logp=np.zeros_like(probability);np.log(probability,out=logp,where=probability>0)
    entropy=-(probability*logp).sum(axis=1).reshape(v.shape)/np.log(120.)
    channel=np.median(np.stack([entropy,mobility,complexity],axis=-1),axis=0)
    answer=np.concatenate([np.median(channel[idx],axis=0) for idx in REGIONS])
    assert answer.shape==(9,) and np.isfinite(answer).all()
    return answer

def phase_surrogate(windows,seed):
    x=np.asarray(windows,float)
    x=x-x.mean(axis=1,keepdims=True) # constant centering is invariant for all three E4 measures.
    spectrum=np.fft.rfft(x,axis=1)
    random=np.random.default_rng(seed)
    transformed=spectrum.copy()
    transformed[:,1:-1,:]=np.abs(spectrum[:,1:-1,:])*np.exp(1j*random.uniform(-np.pi,np.pi,spectrum[:,1:-1,:].shape))
    y=np.fft.irfft(transformed,n=256,axis=1)
    err=float(np.max(np.abs(np.abs(np.fft.rfft(y,axis=1))-np.abs(spectrum)))/max(float(np.max(np.abs(spectrum))),1e-30))
    assert err<=1e-10
    return y,err

def seed_for(pid,index):
    return int.from_bytes(hashlib.sha256(f"20260920|{pid}|{index}".encode()).digest()[:8],"little")

def synthetic_check():
    rng=np.random.default_rng(73912)
    errors=[]
    for x in (rng.normal(size=(5,256,14)),np.round(rng.normal(size=(5,256,14)),1)):
        frozen_values=BATTERY.e4_extract(list(x))
        frozen=np.array([frozen_values[n] for n in NAMES])
        fast=e4_vectorized(x)
        errors.append(float(np.max(np.abs(frozen-fast))))
        assert np.max(np.abs(frozen-fast))<1e-10
        a,e=phase_surrogate(x,981)
        b,_=phase_surrogate(x,981)
        assert np.array_equal(a,b)
    core.write_json(OUT/"synthetic_checks.json",dict(e4_max_errors=errors,determinism=True,spectrum_check=True))

def participant_job(payload):
    pid,record,age,expected_e0,expected_e4,expected_qc=payload
    core.threadpool_limits(1)
    path=RAW.joinpath(*record["eeg_source_file"].replace(chr(92),"/").split("/")[1:]).resolve()
    assert path.is_relative_to(RAW.resolve()) and path.exists()
    source_hash=core.digest(path)
    captured=BATTERY.stage1_capture(STAGE,dict(participant_id=pid,session_id=record["session_id"],eeg_path=path),{pid:{"age":age}})
    qc=captured["qc"]
    assert qc["preprocessing_status"]=="PASS",(pid,qc)
    for name in ("candidate_window_count","rejected_window_count","clean_window_fraction","clean_seconds","bad_channel_count"):
        assert abs(float(qc[name])-float(expected_qc[name]))<1e-10,(pid,name)
    e0=np.array([captured["e0"][n] for n in STAGE.EEG_FEATURES])
    e0_error=float(np.max(np.abs(e0-expected_e0)))
    assert e0_error<1e-10,(pid,e0_error)
    windows=np.stack(captured["accepted_windows"])
    observed=e4_vectorized(windows)
    error=float(np.max(np.abs(observed-expected_e4)))
    assert error<1e-10,(pid,error)
    # A full frozen-function check on each observed participant protects aggregation semantics.
    direct=BATTERY.e4_extract(list(windows))
    direct_error=float(np.max(np.abs(observed-np.array([direct[n] for n in NAMES]))))
    assert direct_error<1e-10
    features=[];errors=[];seeds=[]
    for r in range(20):
        seed=seed_for(pid,r);surrogate,err=phase_surrogate(windows,seed)
        features.append(e4_vectorized(surrogate));errors.append(err);seeds.append(seed)
    assert source_hash==core.digest(path),"Raw file changed during extraction"
    return pid,np.stack(features),dict(participant_id=pid,accepted_windows=len(windows),e0_error=e0_error,
        e4_error=error,vectorized_error=direct_error,spectrum_relative_error=max(errors),raw_sha256=source_hash,
        raw_relative_path=str(path.relative_to(RAW)),seeds=seeds)

def extract(jobs):
    synthetic_check()
    matrix=pd.read_csv(core.AUDIT/"stage1_blind_feature_extraction/BLIND_PRIMARY_FEATURE_MATRIX.csv").set_index("participant_id")
    roster=pd.read_csv(core.AUDIT/"stage1_blind_feature_extraction/BLIND_COHORT_MANIFEST.csv").set_index("participant_id")
    e4=pd.read_csv(core.AUDIT/"exploratory_eeg_battery/E4_RAW_FEATURES.csv").set_index("participant_id")
    qc=pd.read_csv(core.AUDIT/"stage1_blind_feature_extraction/EEG_PREPROCESSING_QC.csv").set_index("participant_id")
    ids=matrix.index.tolist();array=np.full((20,96,9),np.nan);records=[]
    payloads=[(pid,roster.loc[pid].to_dict(),float(matrix.loc[pid,"age"]),matrix.loc[pid,STAGE.EEG_FEATURES].to_numpy(float),e4.loc[pid,NAMES].to_numpy(float),qc.loc[pid].to_dict()) for pid in ids]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for count,result in enumerate(pool.map(participant_job,payloads),1):
            pid,values,record=result
            array[:,ids.index(pid),:]=values;records.append(record)
            print(f"Raw reconstruction + 20 phase controls: {count}/96",flush=True)
    assert np.isfinite(array).all()
    np.savez_compressed(OUT/"surrogate_features.npz",participant_id=np.array(ids),feature_names=np.array(NAMES),features=array)
    core.write_json(OUT/"reconstruction_and_seeds.json",records)
    pd.DataFrame([{k:v for k,v in r.items() if k!="seeds"} for r in records]).to_csv(OUT/"reconstruction_checks.csv",index=False,float_format="%.17g")
    source=e4.loc[ids,NAMES].to_numpy()
    differences=[]
    for j,name in enumerate(NAMES):
        a=array[:,:,j].mean(axis=0);b=source[:,j]
        differences.append(dict(feature=name,observed_mean=float(b.mean()),surrogate_mean=float(a.mean()),
            mean_absolute_shift=float(np.mean(np.abs(a-b))),spearman_observed_vs_surrogate_mean=float(core.scipy.stats.spearmanr(a,b).statistic)))
    pd.DataFrame(differences).to_csv(OUT/"surrogate_feature_diagnostics.csv",index=False,float_format="%.17g")

SIGNAL_DATA=None
def initialize_eval(data):
    global SIGNAL_DATA
    SIGNAL_DATA=data
    core.threadpool_limits(1)

def evaluate_seed(seed_index):
    data,features=SIGNAL_DATA
    raw=np.column_stack([data["raw"]["m2"],features[seed_index]])
    rows=[];selections=[];retries=0
    for outer in data["folds"]["outer"]:
        designs=[]
        for inner in outer["inner_folds"]:
            tr,te=inner["inner_train"],inner["inner_valid"]
            sc=StandardScaler().fit(raw[tr]);designs.append((sc.transform(raw[tr]),sc.transform(raw[te]),tr,te))
        scores={}
        for c in core.GRID:
            losses=[]
            for x,v,tr,te in designs:
                model,retry=core.fit(x,data["y"][tr],c);retries+=int(retry)
                losses.extend(core.loss(data["y"][te],model.predict_proba(v)[:,1]).tolist())
            scores[c]=float(np.mean(losses))
        best=min(core.GRID,key=lambda c:(round(scores[c],12),c))
        tr,te=outer["outer_train"],outer["outer_test"]
        sc=StandardScaler().fit(raw[tr]);model,retry=core.fit(sc.transform(raw[tr]),data["y"][tr],best);retries+=int(retry)
        p=model.predict_proba(sc.transform(raw[te]))[:,1]
        for idx,prob in zip(te,p):rows.append(dict(model=f"surrogate_{seed_index:02d}",surrogate=seed_index,repeat=outer["repeat"],outer_fold=outer["outer_fold"],participant_id=data["ids"][idx],y=int(data["y"][idx]),p=float(prob),selected_C=best))
        for c in core.GRID:selections.append(dict(surrogate=seed_index,repeat=outer["repeat"],outer_fold=outer["outer_fold"],C=c,inner_logloss=scores[c],selected=c==best))
    return rows,selections,retries

def evaluate(jobs):
    data=core.load_data()
    with np.load(OUT/"surrogate_features.npz",allow_pickle=False) as f:
        assert f["participant_id"].tolist()==data["ids"] and f["feature_names"].tolist()==NAMES
        features=f["features"]
    rows=[];selection=[];retries=0
    with ProcessPoolExecutor(max_workers=jobs,initializer=initialize_eval,initargs=((data,features),)) as pool:
        futures=[pool.submit(evaluate_seed,r) for r in range(20)]
        for count,future in enumerate(as_completed(futures),1):
            p,s,r=future.result();rows.extend(p);selection.extend(s);retries+=r
            print(f"Nested evaluation of phase control {count}/20",flush=True)
    pred=pd.DataFrame(rows).sort_values(["surrogate","repeat","outer_fold","participant_id"])
    original=pd.read_csv(core.RESULT/"outer_predictions.csv")
    original=original[original.model.isin(["M2","E4"])];combined=pd.concat([original,pred],ignore_index=True)
    comparisons=[]
    for r in range(20):
        name=f"surrogate_{r:02d}"
        comparisons.extend([(f"M2_vs_{name}","M2",name),(f"{name}_vs_observed_E4",name,"E4")])
    rr,ss=core.summarize_predictions(combined,comparisons)
    pred.to_csv(OUT/"outer_predictions.csv",index=False,float_format="%.17g")
    pd.DataFrame(selection).to_csv(OUT/"inner_selection.csv",index=False,float_format="%.17g")
    rr.to_csv(OUT/"repeat_results.csv",index=False,float_format="%.17g");ss.to_csv(OUT/"comparison_summary.csv",index=False,float_format="%.17g")
    core.write_json(OUT/"RUN_COMPLETE.json",dict(completed_utc=datetime.now(timezone.utc).isoformat(),convergence_retries=retries,
        script_sha256=core.digest(__file__),features_sha256=core.digest(OUT/"surrogate_features.npz"),release_manifest_sha256=core.digest(core.ROOT/"RELEASE_MANIFEST.json")))
    print(ss[["comparison","delta_ll","positive_repeats"]].to_string(index=False),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--phase",choices=["extract","evaluate","all"],default="all");parser.add_argument("--jobs",type=int,default=4)
    args=parser.parse_args();OUT.mkdir(exist_ok=True);core.freeze_sources()
    if args.phase in ("extract","all"):extract(args.jobs)
    if args.phase in ("evaluate","all"):evaluate(args.jobs)
    core.freeze_sources()
if __name__=="__main__":main()
