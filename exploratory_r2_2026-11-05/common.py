"""Shared, training-only exploratory evaluation and fixed-prediction summaries."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import warnings
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
REPO=HERE.parent
GRID=[.01,.1,1.,10.,100.]
B=['age','gender','mean_work_speed','total_omissions','total_commissions',
   'gaze_location_entropy','gaze_spatial_dispersion']
threadpool_limits(1)

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

def read(path):
    return pd.read_csv(path,float_precision='round_trip')

def loss(y,p):
    p=np.clip(np.asarray(p,float),1e-8,1-1e-8)
    y=np.asarray(y,float)
    return -y*np.log(p)-(1-y)*np.log1p(-p)

def fit_predict(x,y,z,c):
    for limit in (1000,10000):
        model=LogisticRegression(C=c,solver='lbfgs',tol=1e-6,max_iter=limit,
                                 class_weight=None,fit_intercept=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');model.fit(x,y)
        bad=any(issubclass(w.category,ConvergenceWarning) for w in caught)
        if bad and limit==1000: continue
        if bad: raise RuntimeError('Logistic head failed to converge after retry')
        p=model.predict_proba(z)[:,1]
        assert np.isfinite(p).all()
        return p

def scale(x,z):
    s=StandardScaler().fit(x)
    return s.transform(x),s.transform(z)

def baseline_data():
    audit=REPO/'audit'
    base=read(audit/'stage1_blind_feature_extraction/BLIND_PRIMARY_FEATURE_MATRIX.csv').set_index('participant_id')
    linked=read(audit/'preregistered_analysis/OUTCOME_LINKAGE_MANIFEST.csv').set_index('participant_id').loc[base.index]
    qc=read(audit/'stage1_blind_feature_extraction/EEG_PREPROCESSING_QC.csv').set_index('participant_id').loc[base.index]
    e4=read(audit/'exploratory_eeg_battery/E4_RAW_FEATURES.csv').set_index('participant_id').loc[base.index]
    assert len(base)==96 and linked.index.equals(base.index)
    nuisance=np.column_stack([qc.device.eq('Epoc+').astype(float),qc.clean_window_fraction,qc.interpolation_count])
    assert np.isfinite(nuisance).all()
    return base,linked,e4,nuisance

def fold_records(manifest,ids):
    lookup={pid:i for i,pid in enumerate(ids)}
    for (r,f),frame in manifest.groupby(['repeat','outer_fold'],sort=True):
        def take(level,role,inner=None):
            d=frame[frame.level.eq(level)&frame.role.eq(role)]
            if inner is not None: d=d[d.inner_fold.eq(inner)]
            d=d.sort_values('index')
            return np.array([lookup[p] for p in d.participant_id],int)
        tr,te=take('outer','train'),take('outer','test')
        assert not set(tr)&set(te) and set(tr)|set(te)==set(range(len(ids)))
        folds=[];valid=[]
        for j in range(5):
            it,iv=take('inner','train',j),take('inner','valid',j)
            assert not set(it)&set(iv) and set(it)|set(iv)==set(tr)
            assert not (set(it)|set(iv))&set(te)
            valid.extend(iv);folds.append((it,iv))
        assert sorted(valid)==sorted(tr)
        yield int(r),int(f),tr,te,folds

def nested(ids,y,manifest,designs,out):
    out.mkdir(parents=True,exist_ok=True)
    predictions=[];tuning=[]
    for r,f,tr,te,folds in fold_records(manifest,ids):
        pair={}
        for name,design in designs.items():
            cached=[(*design(it,iv),it,iv) for it,iv in folds]
            scores={}
            for c in GRID:
                losses=[]
                for x,z,it,iv in cached:
                    p=fit_predict(x,y[it],z,c)
                    losses.extend(loss(y[iv],p))
                scores[c]=float(np.mean(losses))
            best=min(GRID,key=lambda c:(round(scores[c],12),c))
            for c in GRID:
                tuning.append(dict(repeat=r,outer_fold=f,model=name,C=c,
                                   inner_logloss=scores[c],selected=c==best))
            x,z=design(tr,te)
            pair[name]=fit_predict(x,y[tr],z,best)
        for j,idx in enumerate(te):
            predictions.append(dict(repeat=r,outer_fold=f,participant_id=ids[idx],y=int(y[idx]),
                                    **{'p_'+n:float(p[j]) for n,p in pair.items()}))
        if f==4:
            print(f'{out.parent.name}/{out.name}: repeat {r+1}/10',flush=True)
        pd.DataFrame(predictions).to_csv(out/'OUTER_PREDICTIONS.csv',index=False,float_format='%.17g')
    pd.DataFrame(tuning).to_csv(out/'INNER_SELECTION.csv',index=False,float_format='%.17g')
    return pd.DataFrame(predictions)

def summarize(pred,baseline,augmented,name,out,seed=20261105):
    out.mkdir(parents=True,exist_ok=True)
    assert not pred.duplicated(['repeat','participant_id']).any()
    rows=[]
    for r,g in pred.groupby('repeat'):
        b=g[baseline].to_numpy();a=g[augmented].to_numpy();y=g.y.to_numpy()
        bl=float(loss(y,b).mean());al=float(loss(y,a).mean())
        rows.append(dict(comparison=name,repeat=int(r),baseline_logloss=bl,augmented_logloss=al,
                         delta_ll=bl-al,delta_auroc=float(roc_auc_score(y,a)-roc_auc_score(y,b)),
                         delta_brier=float(np.mean((y-b)**2)-np.mean((y-a)**2))))
    rr=pd.DataFrame(rows)
    per=pred[['participant_id','y']].copy()
    per['delta_ll']=loss(pred.y,pred[baseline])-loss(pred.y,pred[augmented])
    per=per.groupby('participant_id',sort=True).agg(y=('y','first'),delta_ll=('delta_ll','mean'))
    assert pred.groupby('participant_id').y.nunique().eq(1).all()
    rng=np.random.default_rng(seed)
    classes=[np.flatnonzero(per.y.to_numpy()==label) for label in [0,1]]
    draws=[]
    for _ in range(10000):
        sampled=np.concatenate([rng.choice(ix,len(ix),replace=True) for ix in classes])
        draws.append(float(per.delta_ll.to_numpy()[sampled].mean()))
    lower,upper=np.quantile(draws,[.025,.975],method='linear')
    result=dict(comparison=name,participants=len(per),positive_labels=int(per.y.sum()),
                **{k:float(rr[k].mean()) for k in
                   ['baseline_logloss','augmented_logloss','delta_ll','delta_auroc','delta_brier']},
                positive_repeats=int(rr.delta_ll.gt(0).sum()),lower=float(lower),upper=float(upper))
    rr.to_csv(out/(name+'_REPEATS.csv'),index=False,float_format='%.17g')
    per.to_csv(out/(name+'_PARTICIPANT_LOSSES.csv'),float_format='%.17g')
    pd.DataFrame({'draw':np.arange(10000),'delta_ll':draws}).to_csv(
        out/(name+'_CONDITIONAL_DRAWS.csv'),index=False,float_format='%.17g')
    return result

def provenance(out,inputs,extra=None):
    write(out/'PROVENANCE.json',dict(status='complete',analysis_status='exploratory',
        completed_utc=datetime.now(timezone.utc).isoformat(),numpy=np.__version__,
        pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,
        input_sha256={str(p):sha(p) for p in inputs},common_code_sha256=sha(__file__),
        conditional_range='10000 class-stratified participant resamples of mean paired LL; predictions fixed',
        **(extra or {})))
