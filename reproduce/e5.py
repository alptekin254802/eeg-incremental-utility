"""Evaluate post-hoc E5 with band-specific dimensionality reduction."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import importlib.util, json, hashlib
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent; PUBLIC=Path(os.environ['EEG_RELEASE_ROOT']); OUT=Path(os.environ['EEG_RUN_ROOT'])/'e5'
def load(relative,name):
    spec=importlib.util.spec_from_file_location(name,PUBLIC/relative)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
primary=load('audit/preregistered_analysis/run_preregistered_analysis.py','e5_primary')
battery=load('audit/exploratory_eeg_battery/exploratory_eeg_battery.py','e5_battery')
def write_json(p,obj): p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
DATA=None
def init(data):
    global DATA
    DATA=data;threadpool_limits(1)
def worker(index):
    phase,allfolds,predictions=DATA
    folds=dict(allfolds,outer=[allfolds['outer'][index]])
    errors=[]
    pred,fold,sel=battery.run_family('E5',phase,primary,predictions,folds,errors)
    return pred,fold,sel,errors
def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'RESULTS.json').exists(): raise RuntimeError('Completed E5 run exists; use a new output directory')
    base,measurement=primary.load_measurements()
    ids=measurement['ids']
    features=pd.read_csv(PUBLIC/'audit/exploratory_eeg_battery/E5_RAW_FEATURES.csv').set_index('participant_id').loc[ids]
    quantities=('occupancy','median_high_state_duration_seconds','switching_rate_per_minute')
    columns=[f'{region}_{band}_{q}' for band in battery.BAND_ORDER for region in battery.REGIONS for q in quantities]
    assert len(columns)==len(set(columns))==27 and set(columns)==set(features.columns)
    blocks=[]
    for j,band in enumerate(battery.BAND_ORDER):
        names=columns[j*9:(j+1)*9]
        assert len(names)==9 and all(f'_{band}_' in n for n in names)
        assert all(sum(n.startswith(region+'_') for n in names)==3 for region in battery.REGIONS)
        blocks.append(dict(band=band,columns=names))
    band_features=features.loc[:,columns].reset_index()
    saved=pd.read_csv(PUBLIC/'audit/preregistered_analysis/PRIMARY_OUTER_PREDICTIONS.csv')
    y=saved[saved.repeat==0].set_index('participant_id').loc[ids,'y'].to_numpy(int)
    assert len(ids)==96 and int(y.sum())==41
    for _,g in saved.groupby('repeat'): assert np.array_equal(g.set_index('participant_id').loc[ids,'y'],y)
    manifest=pd.read_csv(PUBLIC/'audit/preregistered_analysis/PRIMARY_FOLD_MANIFEST.csv')
    folds=battery.compare_outer_assignments(ids,y,saved,manifest,primary)
    membership_checks=0;rankrows=[]
    x=band_features.drop(columns='participant_id').to_numpy(float)
    for outer in folds['outer']:
        partitions=[('outer',-1,outer['outer_train'])]+[('inner',i['inner_fold'],i['inner_train']) for i in outer['inner_folds']]
        for inner in outer['inner_folds']:
            m=manifest[(manifest.level=='inner')&(manifest.repeat==outer['repeat'])&(manifest.outer_fold==outer['outer_fold'])&(manifest.inner_fold==inner['inner_fold'])]
            for role,idx in [('train',inner['inner_train']),('valid',inner['inner_valid'])]:
                assert set(m[m.role==role].participant_id)==set(np.array(ids)[idx]);membership_checks+=1
        for level,index,tr in partitions:
            for b,band in enumerate(battery.BAND_ORDER):
                z=x[tr,b*9:b*9+9];rank=int(np.linalg.matrix_rank(z-z.mean(axis=0)))
                assert rank>=3
                rankrows.append(dict(repeat=outer['repeat'],outer_fold=outer['outer_fold'],level=level,inner_fold=index,band=band,rank=rank))
    pd.DataFrame(rankrows).to_csv(OUT/'BAND_RANK_CHECKS.csv',index=False)
    band_features.to_csv(OUT/'E5_BAND_ORDERED_FEATURES.csv',index=False,float_format='%.17g')
    inputs=json.loads((PUBLIC/'RELEASE_MANIFEST.json').read_text(encoding='utf-8'))['sha256']
    assert all(sha(PUBLIC/p)==h for p,h in inputs.items())
    contract=dict(created_utc=datetime.now(timezone.utc).isoformat(),
       status='Post-hoc exploratory analysis',
       method='Group 27 predictors by frequency band; standardize and fit three principal components per band within training partitions.',
       n=96,positive=41,outer_fits=50,inner_partitions=250,inner_membership_checks=membership_checks,
       bands=blocks,C_grid=battery.C_GRID,pc_per_band=3,raw_sha256=sha(PUBLIC/'audit/exploratory_eeg_battery/E5_RAW_FEATURES.csv'),
       estimator_code_sha256=sha(PUBLIC/'audit/exploratory_eeg_battery/exploratory_eeg_battery.py'),specification='Supplementary Section S5.3: post-hoc E5 band grouping',inputs=inputs)
    if not (OUT/'ANALYSIS_SPECIFICATION.json').exists(): write_json(OUT/'ANALYSIS_SPECIFICATION.json',contract)
    else:
        existing=json.loads((OUT/'ANALYSIS_SPECIFICATION.json').read_text(encoding='utf-8'))
        assert existing['inputs']==inputs and existing['bands']==blocks
    phase={'ids':ids,'e5_frame':band_features}
    predictions=[];foldrows=[];selection=[];failures=[]
    print('Running E5 over all 50 outer fits.',flush=True)
    with threadpool_limits(1):
        predictions,foldrows,selection=battery.run_family('E5',phase,primary,saved,folds,failures)
    assert len(predictions)==960 and len(foldrows)==50 and len(selection)==250
    pred=pd.DataFrame(predictions).sort_values(['repeat','participant_index'])
    sel=pd.DataFrame(selection).sort_values(['repeat','outer_fold','C'])
    pred.to_csv(OUT/'OUTER_PREDICTIONS.csv',index=False,float_format='%.17g')
    sel.to_csv(OUT/'INNER_SELECTION.csv',index=False,float_format='%.17g')
    pd.DataFrame(foldrows).sort_values(['repeat','outer_fold']).to_csv(OUT/'FOLD_RESULTS.csv',index=False,float_format='%.17g')
    write_json(OUT/'FAILURES.json',failures)
    rr=[]
    for r,g in pred.groupby('repeat'):
        yy=g.y.to_numpy(int); p2=g.p_m2.to_numpy(float);pe=g.p_eeg.to_numpy(float)
        def metrics(p):
            pc=np.clip(p,1e-8,1-1e-8);d=p[yy==1,None]-p[None,yy==0]
            return dict(logloss=float(np.mean(-yy*np.log(pc)-(1-yy)*np.log1p(-pc))),auroc=float(np.mean((d>0)+.5*(d==0))),brier=float(np.mean((yy-p)**2)))
        a,b=metrics(p2),metrics(pe)
        rr.append(dict(family='E5',repeat=int(r),n=len(g),**{'m2_'+k:v for k,v in a.items()},**{'eeg_'+k:v for k,v in b.items()},
                       delta_ll=a['logloss']-b['logloss'],delta_auroc=b['auroc']-a['auroc'],delta_brier=a['brier']-b['brier']))
    repeat=pd.DataFrame(rr);repeat.to_csv(OUT/'REPEAT_RESULTS.csv',index=False,float_format='%.17g')
    for (r,f),g in sel.groupby(['repeat','outer_fold']):
        chosen=min(g[g.valid].itertuples(),key=lambda a:(round(a.pooled_inner_logloss,12),a.C)).C
        assert g[g.selected].C.tolist()==[chosen]
        assert pred[(pred.repeat==r)&(pred.outer_fold==f)].selected_C.unique().tolist()==[chosen]
    summary=dict(completed_utc=datetime.now(timezone.utc).isoformat(),n=96,outer_fits=50,inner_partitions=250,
        chosen_C_counts={str(c):int(v) for c,v in sel[sel.selected].C.value_counts().items()},failure_events=len(failures),
        metrics={k:float(repeat[k].mean()) for k in repeat if k not in ('family','repeat','n')},
        positive_repeats=int((repeat.delta_ll>0).sum()),range_delta_ll=[float(repeat.delta_ll.min()),float(repeat.delta_ll.max())],
        input_preservation=all(sha(PUBLIC/p)==h for p,h in inputs.items()))
    assert summary['input_preservation'] and not failures
    write_json(OUT/'RESULTS.json',summary)
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__': main()
