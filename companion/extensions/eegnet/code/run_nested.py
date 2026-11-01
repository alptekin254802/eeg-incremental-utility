"""One fixed, fully nested EEGNet-like representation comparison (exploratory)."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse, json, sys, time, warnings, platform
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch, sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from eegnet_model import Encoder, train_encoder, embeddings
from prepare_windows import sha, write, cache_fingerprint
sys.dont_write_bytecode=True
GRID=[.01,.1,1.,10.,100.]
B=['age','gender','mean_work_speed','total_omissions','total_commissions','gaze_location_entropy','gaze_spatial_dispersion']

def loss(y,p):
    p=np.clip(p,1e-8,1-1e-8)
    return -y*np.log(p)-(1-y)*np.log1p(-p)

def fit_head(x,y,z,c):
    scale=StandardScaler().fit(x)
    for limit in (1000,10000):
        m=LogisticRegression(C=c,solver='lbfgs',tol=1e-6,max_iter=limit,class_weight=None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');m.fit(scale.transform(x),y)
        bad=any(issubclass(w.category,ConvergenceWarning) for w in caught)
        if bad and limit==1000: continue
        if bad:raise RuntimeError('Head did not converge after retry')
        p=m.predict_proba(scale.transform(z))[:,1]
        assert np.isfinite(p).all()
        meta=dict(C=c,mean=scale.mean_.tolist(),scale=scale.scale_.tolist(),coef=m.coef_.tolist(),intercept=m.intercept_.tolist(),iterations=m.n_iter_.tolist(),retry=limit==10000)
        return p,meta

def load_data(release,cache):
    manifest=json.loads((cache/'INPUT_MANIFEST.json').read_text())
    for file,digest in manifest['reference_inputs'].items():
        if sha(release/file)!=digest:raise RuntimeError('Reference mismatch: '+file)
    s=release/'audit'
    base=pd.read_csv(s/'stage1_blind_feature_extraction/BLIND_PRIMARY_FEATURE_MATRIX.csv').set_index('participant_id')
    old=pd.read_csv(s/'preregistered_analysis/PRIMARY_OUTER_PREDICTIONS.csv')
    old=old[old.outcome.eq('group')].copy()
    y=old[old.repeat.eq(0)].set_index('participant_id').loc[base.index,'y'].astype(int)
    folds=pd.read_csv(s/'preregistered_analysis/PRIMARY_FOLD_MANIFEST.csv')
    assert len(base)==96 and sum(y)==41 and len(old)==960
    windows={}
    for item in manifest['participants']:
        if sha(cache/item['cache_file'])!=item['cache_sha256']:raise RuntimeError('Window identity mismatch')
        windows[item['participant_id']]=np.load(cache/item['cache_file'],allow_pickle=False)
    assert set(windows)==set(base.index)
    return base[B],y,folds,old,windows

def partition(frame,level,role,inner=None):
    f=frame[(frame.level==level)&(frame.role==role)]
    if inner is not None:f=f[f.inner_fold.eq(inner)]
    return f.sort_values('index').participant_id.tolist()

def encoder_fit(windows,tr,ev,y,seed,device,out):
    out.mkdir(parents=True,exist_ok=False)
    assert not set(tr)&set(ev)
    start=time.perf_counter()
    model,mean,scale,curves=train_encoder(windows,tr,y.loc[tr].to_numpy(),seed,device=device)
    z=embeddings(model,windows,tr+ev,mean,scale,device=device)
    assert z.shape==(len(tr)+len(ev),128) and np.isfinite(z).all()
    np.savez_compressed(out/'embeddings.npz',participant_id=np.asarray(tr+ev),embedding=z,channel_mean=mean,channel_scale=scale)
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},out/'encoder.pt')
    write(out/'fit.json',dict(seed=seed,train_ids=tr,assessment_ids=ev,epochs=50,loss=curves,seconds=time.perf_counter()-start,device=device,parameters=sum(p.numel() for p in model.parameters())))
    return z[:len(tr)],z[len(tr):]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--release',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--lock',type=Path,required=True);ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');ap.add_argument('--fold',type=int,default=None);a=ap.parse_args()
    lock=json.loads(a.lock.read_text())
    for name,digest in lock['scientific_files'].items():
        if sha(a.lock.parent/name)!=digest:raise RuntimeError('Locked code/plan changed: '+name)
    if cache_fingerprint(json.loads((a.cache/'INPUT_MANIFEST.json').read_text()))!=lock['window_data_fingerprint']:raise RuntimeError('Wrong cache data/configuration')
    if a.out.exists():raise FileExistsError('Use a new output directory')
    a.out.mkdir(parents=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);threadpool_limits(1)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    base,y,folds,old,windows=load_data(a.release,a.cache)
    write(a.out/'RUN.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),plan_lock_sha256=sha(a.lock),window_manifest_sha256=sha(a.cache/'INPUT_MANIFEST.json'),python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,sklearn=sklearn.__version__,device=a.device,device_name=torch.cuda.get_device_name(0) if a.device=='cuda' else 'CPU',scope='50 outer partitions' if a.fold is None else 'selected same-seed replay',selected_fold=a.fold))
    predictions=[];candidates=[]
    for task in range(50) if a.fold is None else [a.fold]:
        repeat,outer=divmod(task,5);frame=folds[folds.repeat.eq(repeat)&folds.outer_fold.eq(outer)]
        tr=partition(frame,'outer','train');te=partition(frame,'outer','test')
        assert len(set(tr)|set(te))==96 and not set(tr)&set(te)
        foldout=a.out/f'r{repeat:02d}_f{outer:02d}';foldout.mkdir()
        acc={k:{c:[] for c in GRID} for k in ('baseline','eegnet')}
        validation_ids=[]
        for inner in range(5):
            it=partition(frame,'inner','train',inner);iv=partition(frame,'inner','valid',inner)
            assert set(it)|set(iv)==set(tr) and not set(it)&set(iv)
            assert not (set(it)|set(iv))&set(te)
            validation_ids.extend(iv)
            seed=20261101+10000*repeat+100*outer+inner+1
            zt,zv=encoder_fit(windows,it,iv,y,seed,a.device,foldout/f'inner{inner}')
            designs={'baseline':(base.loc[it].to_numpy(),base.loc[iv].to_numpy()),'eegnet':(np.column_stack([base.loc[it],zt]),np.column_stack([base.loc[iv],zv]))}
            for kind,(x,z) in designs.items():
                for c in GRID:
                    p,meta=fit_head(x,y.loc[it].to_numpy(),z,c)
                    acc[kind][c].extend(loss(y.loc[iv].to_numpy(),p).tolist())
                    write(foldout/f'inner{inner}'/f'{kind}_C{c:g}.json',dict(head=meta,validation_ids=iv,probabilities=p.tolist()))
            print(f'completed encoder {task*6+inner+1}/300',flush=True)
        assert len(validation_ids)==len(tr) and set(validation_ids)==set(tr)
        selected={}
        for kind in acc:
            scores={c:float(np.mean(acc[kind][c])) for c in GRID}
            selected[kind]=min(GRID,key=lambda c:(round(scores[c],12),GRID.index(c)))
            for c in GRID:candidates.append(dict(repeat=repeat,outer_fold=outer,model=kind,C=c,inner_logloss=scores[c],selected=c==selected[kind]))
        zt,zv=encoder_fit(windows,tr,te,y,20261101+10000*repeat+100*outer,a.device,foldout/'outer')
        both={}
        for kind,(x,z) in {'baseline':(base.loc[tr].to_numpy(),base.loc[te].to_numpy()),'eegnet':(np.column_stack([base.loc[tr],zt]),np.column_stack([base.loc[te],zv]))}.items():
            both[kind],meta=fit_head(x,y.loc[tr].to_numpy(),z,selected[kind])
            write(foldout/'outer'/f'{kind}_head.json',meta)
        expected=old[old.repeat.eq(repeat)&old.outer_fold.eq(outer)].set_index('participant_id').loc[te]
        error=float(np.max(np.abs(both['baseline']-expected.p_m2.to_numpy())))
        if error>1e-7:raise RuntimeError(f'Baseline replay mismatch {error}')
        for j,pid in enumerate(te):predictions.append(dict(repeat=repeat,outer_fold=outer,participant_id=pid,y=int(y.loc[pid]),p_baseline=both['baseline'][j],p_eegnet=both['eegnet'][j],C_baseline=selected['baseline'],C_eegnet=selected['eegnet']))
        write(foldout/'COMPLETE.json',dict(baseline_max_abs_error=error,selected=selected,encoders=6))
        pd.DataFrame(predictions).to_csv(a.out/'OUTER_PREDICTIONS.csv',index=False,float_format='%.17g')
        pd.DataFrame(candidates).to_csv(a.out/'INNER_CANDIDATES.csv',index=False,float_format='%.17g')
        print(f'completed outer {task+1}/50; baseline identity verified',flush=True)
    write(a.out/'COMPLETE.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),outer_partitions=50 if a.fold is None else 1,prediction_rows=len(predictions),encoder_fits=300 if a.fold is None else 6))

if __name__=='__main__':main()
