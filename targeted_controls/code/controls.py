"""Bounded exploratory controls. Inputs are read-only; each output stage is new."""
import os,sys
sys.dont_write_bytecode=True
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,hashlib,importlib.util,itertools,time,platform
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,pandas as pd,scipy,sklearn
from scipy import signal
from scipy.special import expit,logit
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
threadpool_limits(1)
CORE=None;DATA=None;ROOT=None;RELEASE=None
GRID=[.01,.1,1.,10.,100.]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def csv(p,rows):pd.DataFrame(rows).to_csv(p,index=False,float_format='%.17g')
def load(p,name):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def loss(y,p):
    p=np.clip(np.asarray(p),1e-8,1-1e-8);y=np.asarray(y);return -y*np.log(p)-(1-y)*np.log1p(-p)
def metrics(y,p):return dict(logloss=float(loss(y,p).mean()),auroc=float(roc_auc_score(y,p)),brier=float(np.mean((np.asarray(y)-p)**2)))
def init(release,out,data=None):
    global CORE,DATA,ROOT,RELEASE
    ROOT=Path(out);RELEASE=Path(release);threadpool_limits(1)
    os.environ['EEG_RELEASE_ROOT']=str(RELEASE);os.environ['EEG_RUN_ROOT']=str(ROOT/'runtime')
    CORE=load(RELEASE/'reproduce/run_posthoc.py','frozen_posthoc');DATA=data
def spline(x):
    x=np.asarray(x,float);k=np.array([6.,10.,14.,18.]);out=[x]
    for j in range(2):
        h=np.maximum(x-k[j],0)**3-np.maximum(x-k[2],0)**3*(k[3]-k[j])/(k[3]-k[2])+np.maximum(x-k[3],0)**3*(k[2]-k[j])/(k[3]-k[2])
        out.append(h/(k[3]-k[0])**2)
    return np.column_stack(out)
def age_data(data):
    r={**data,'raw':dict(data['raw'])};m=data['raw']['m2'];r['raw']['m2']=np.column_stack([spline(m[:,0]),m[:,1:]])
    assert r['raw']['m2'].shape==(96,9);return r
def tests(out):
    x=np.array([2.,3.,4.,20.,21.,22.]);a=spline(x)
    assert np.max(np.abs(np.diff(a[:3],n=2,axis=0)))<1e-12
    assert np.max(np.abs(np.diff(a[3:],n=2,axis=0)))<1e-12
    for k in [6.,10.,14.,18.]:
        eps=1e-5;v=spline([k-eps,k,k+eps]);assert np.max(np.abs((v[2]-v[1])-(v[1]-v[0])))<1e-8
    p=np.array([.1,.5,.9]);s=.3;t=.6;d=logit(t)-logit(s)
    odds=(p/(1-p))*(t/(1-t))/(s/(1-s));assert np.max(np.abs(expit(logit(p)+d)-odds/(1+odds)))<1e-15
    assert np.array_equal(np.argsort(p),np.argsort(expit(logit(p)+d)))
    sos=signal.butter(4,30,fs=128,btype='lowpass',output='sos');freq,h=signal.sosfreqz(sos,worN=np.array([10.,30.,40.]),fs=128)
    assert abs(h[2])<abs(h[1])<abs(h[0]);assert np.isclose(abs(h[1])**2,.5)
    dump(out/'SYNTHETIC_CHECKS.json',dict(status='PASS',spline_fixed_knots=[6,10,14,18],spline_continuity_and_linear_tails=True,prior_odds_algebra=True,filter_single_pass_gain=np.abs(h).tolist(),note='Zero-phase forward/backward filtering squares the single-pass magnitude.'))
def e4(windows,battery):
    x=np.asarray(windows,float).transpose(0,2,1);med=np.median(x,axis=2,keepdims=True);scale=1.4826*np.median(np.abs(x-med),axis=2,keepdims=True)
    assert np.isfinite(x).all() and (scale>0).all();x=(x-med)/scale
    v=np.var(x,axis=2);v1=np.var(np.diff(x,axis=2),axis=2);v2=np.var(np.diff(x,n=2,axis=2),axis=2)
    mob=np.sqrt(v1/v);comp=np.sqrt(v2/v1)/mob
    lut=np.full(3125,-1,dtype=int)
    for i,p in enumerate(itertools.permutations(range(5))):lut[np.ravel_multi_index(p,(5,)*5)]=i
    orders=np.argsort(np.lib.stride_tricks.sliding_window_view(x,5,axis=2),axis=-1,kind='stable');codes=(orders*np.array([625,125,25,5,1])).sum(axis=-1)
    ranks=lut[codes].reshape(-1,252);assert (ranks>=0).all();offset=np.arange(len(ranks))[:,None]*120
    counts=np.bincount((ranks+offset).ravel(),minlength=len(ranks)*120).reshape(-1,120);p=counts/252;lp=np.zeros_like(p);np.log(p,out=lp,where=p>0)
    ent=-(p*lp).sum(axis=1).reshape(v.shape)/np.log(120)
    chan=np.median(np.stack([ent,mob,comp],axis=-1),axis=0)
    vals=np.concatenate([np.median(chan[[battery.CHANNELS.index(c) for c in group]],axis=0) for group in battery.REGIONS.values()])
    assert vals.shape==(9,) and np.isfinite(vals).all();return vals
def extract_one(args):
    release,raw,record,expected0,expected4,expectedq=args;release=Path(release);raw=Path(raw)
    stage=load(release/'audit/stage1_blind_feature_extraction/run_stage1_extraction.py','stage');bat=load(release/'audit/exploratory_eeg_battery/exploratory_eeg_battery.py','battery')
    pid=record['participant_id'];rel=Path(*record['eeg_source_file'].replace('\\','/').split('/')[1:]);path=(raw/rel).resolve();assert path.is_relative_to(raw.resolve()) and path.exists()
    h=sha(path);got=bat.stage1_capture(stage,dict(participant_id=pid,session_id=str(record['session_id']),eeg_path=path),{pid:{'age':expected0['age']}})
    assert got['qc']['preprocessing_status']=='PASS'
    for k in ['candidate_window_count','rejected_window_count','clean_window_fraction','clean_seconds','bad_channel_count']:
        assert abs(float(got['qc'][k])-float(expectedq[k]))<1e-8,(pid,k)
    err0=max(abs(got['e0'][k]-expected0[k]) for k in stage.EEG_FEATURES);assert err0<1e-8
    observed=e4(got['accepted_windows'],bat);err4=float(np.max(np.abs(observed-np.asarray(expected4))));assert err4<1e-8,(pid,err4)
    sos=signal.butter(4,30,fs=128,btype='lowpass',output='sos')
    filtered=[signal.sosfiltfilt(sos,seg,axis=0) for seg in got['segments']]
    windows=[filtered[seg][start:start+256] for seg,start in got['accepted_ids']]
    assert len(windows)==len(got['accepted_windows']);new=e4(windows,bat)
    assert sha(path)==h
    return dict(participant_id=pid,features=new.tolist(),original_features=observed.tolist(),raw_relative_path=rel.as_posix(),raw_sha256=h,e0_max_error=err0,e4_max_error=err4,accepted_ids=got['accepted_ids'],segments=len(filtered),accepted_windows=len(windows))
def extract(args,data,out):
    roster=pd.read_csv(RELEASE/'audit/stage1_blind_feature_extraction/BLIND_COHORT_MANIFEST.csv',dtype={'session_id':str}).set_index('participant_id')
    names=data['e4_names'];rows=[];audit=[]
    payload=[]
    for i,pid in enumerate(data['ids']):payload.append((str(RELEASE),str(args.raw),dict(roster.loc[pid],participant_id=pid),data['base'].set_index('participant_id').loc[pid].to_dict(),data['e4'][i],data['qc'].loc[pid].to_dict()))
    start=time.monotonic()
    with ProcessPoolExecutor(max_workers=min(args.workers,2)) as ex:
        for i,r in enumerate(ex.map(extract_one,payload),1):
            csv(out/(r['participant_id']+'_features.csv'),[dict(participant_id=r['participant_id'],**dict(zip(names,r['features'])))])
            rows.append(dict(participant_id=r['participant_id'],**dict(zip(names,r.pop('features')))));r.pop('original_features');audit.append(r)
            if i==1 or i%12==0:print(f'extract {i}/96; {time.monotonic()-start:.0f}s',flush=True)
    csv(out/'E4_LOWPASS30_FEATURES.csv',rows);dump(out/'RAW_RECONSTRUCTION.json',audit)
    merged=pd.DataFrame(rows).set_index('participant_id').loc[data['ids']]
    csv(out/'FEATURE_COMPARISON.csv',[dict(feature=n,spearman=float(spearmanr(data['e4'][:,i],merged[n]).statistic),mean_original=float(data['e4'][:,i].mean()),mean_filtered=float(merged[n].mean())) for i,n in enumerate(names)])
    dump(out/'COMPLETE.json',dict(status='PASS',participants=96,seconds=time.monotonic()-start,all_original_features_reproduced=True))
def fit_one(spec):
    outer,part=spec;rep=outer['repeat'];fold=outer['outer_fold'];d=DATA if part=='bandwidth' else age_data(DATA)
    if part=='age': specs=[('baseline_spline','none',0),('E0_spline','E0',0),('E1_spline','E1',0),('E2_spline','E2',0),('entropy_spline','E4',0),('baseline_spline_DQ','none',3),('entropy_spline_DQ','E4',3)]
    else:specs=[('entropy_lowpass30','E4',0),('entropy_lowpass30_DQ','E4',3)]
    inn=outer['inner_folds'];tr=np.array(outer['outer_train']);te=np.array(outer['outer_test']);ic=[{} for _ in inn];oc={};pred=[];tuning=[];foldscores=[];choice=[];probs={};bestby={};retries=0;fits=0
    # Reference replay once per family, including adjusted/unadjusted baselines.
    replay_specs=[('replay_baseline','none',0),('replay_entropy','E4',0),('replay_baseline_DQ','none',3)] if rep==0 and fold==0 and part=='age' else []
    for name,kind,width in specs+replay_specs:
        dd=DATA if name.startswith('replay') else d
        caches=[{} for _ in inn] if name.startswith('replay') else ic;cache={} if name.startswith('replay') else oc
        mats=[CORE.design(dd,i['inner_train'],i['inner_valid'],kind,width,c)[:2] for i,c in zip(inn,caches)]
        scores=[]
        for c in GRID:
            ls=[]
            for j,(i,(x,v)) in enumerate(zip(inn,mats)):
                m,retry=CORE.fit(x,dd['y'][i['inner_train']],c);retries+=int(retry);fits+=1;p=m.predict_proba(v)[:,1];ll=loss(dd['y'][i['inner_valid']],p);ls.extend(ll.tolist());foldscores.append(dict(model=name,C=c,inner_fold=j,logloss=float(ll.mean()),n=len(ll)))
            scores.append(float(np.mean(ls)))
        idx=min(range(5),key=lambda i:(round(scores[i],12),GRID[i]));c=GRID[idx]
        x,v,_=CORE.design(dd,tr,te,kind,width,cache)
        probe={**dd,'raw':{k:z.copy() for k,z in dd['raw'].items()},'e4':dd['e4'].copy(),'nuisance':dd['nuisance'].copy()}
        for k,z in probe['raw'].items():
            if k=='covariances':z[te]=np.eye(z.shape[-1])[None,None,:,:]/z.shape[-1]
            else:z[te]+=17
        probe['e4'][te]+=17;probe['nuisance'][te]+=17
        px,_,_=CORE.design(probe,tr,te,kind,width,{})
        assert np.allclose(x,px,rtol=0,atol=1e-12),(name,'heldout affected training')
        m,retry=CORE.fit(x,dd['y'][tr],c);fits+=1;retries+=int(retry);pp=m.predict_proba(v)[:,1]
        assert np.isfinite(pp).all();bestby[name]=(c,scores[idx]);probs[name]=pp
        for ci,cc in enumerate(GRID):tuning.append(dict(model=name,C=cc,inner_logloss=scores[ci],selected=ci==idx))
        for i,p in zip(te,pp):pred.append(dict(model=name,participant_id=dd['ids'][i],y=int(dd['y'][i]),p=float(p),C=c))
    if part=='age':
        names=['E0_spline','E1_spline','E2_spline'];chosen=min(names,key=lambda n:(round(bestby[n][1],12),names.index(n),bestby[n][0]));choice=[dict(selected=chosen,C=bestby[chosen][0])]
        for i,p in zip(te,probs[chosen]):pred.append(dict(model='selector_spline',participant_id=d['ids'][i],y=int(d['y'][i]),p=float(p),C=bestby[chosen][0]))
    for rows in [pred,tuning,foldscores,choice]:
        for row in rows:row.update(repeat=rep,outer_fold=fold)
    return dict(predictions=pred,tuning=tuning,inner=foldscores,choices=choice,fit_calls=fits,retries=retries,repeat=rep,fold=fold)
def refit(args,data,out):
    part=args.part
    if part=='bandwidth':
        p=args.output/'extract/E4_LOWPASS30_FEATURES.csv';assert json.loads((p.parent/'COMPLETE.json').read_text())['status']=='PASS'
        data={**data,'e4':pd.read_csv(p,float_precision='round_trip').set_index('participant_id').loc[data['ids'],data['e4_names']].to_numpy()}
    rows={k:[] for k in ['predictions','tuning','inner','choices']};done=[];start=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init,initargs=(str(RELEASE),str(ROOT),data)) as ex:
        futures=[ex.submit(fit_one,(o,part)) for o in data['folds']['outer']]
        for i,f in enumerate(as_completed(futures),1):
            r=f.result();job=out/f"r{r['repeat']:02}_f{r['fold']}";job.mkdir()
            for k in rows:csv(job/(k+'.csv'),r[k]);rows[k].extend(r.pop(k))
            dump(job/'DONE.json',r);done.append(r)
            if i==1 or i%10==0:print(f'{part} {i}/50; {time.monotonic()-start:.0f}s',flush=True)
    for k in rows:csv(out/(k.upper()+'.csv'),rows[k])
    dump(out/'COMPLETE.json',dict(status='PASS',outer_tasks=50,fit_calls=sum(r['fit_calls'] for r in done),convergence_retries=sum(r['retries'] for r in done),heldout_training_invariance=True,seconds=time.monotonic()-start))
def offset(args,data,out):
    p=args.transfer/'analysis/results/PREDICTIONS.csv';frame=pd.read_csv(p,float_precision='round_trip');records=[];contrasts=[];drawrows=[];rng=np.random.default_rng(2026110301)
    for direction,sub in frame.groupby('direction',sort=True):
        base=sub[sub.model=='baseline'].set_index('participant_id').sort_index();y=base.y.to_numpy();pt=float(y.mean());ps=16/54 if direction=='X_to_Plus' else 25/42;offset=float(logit(pt)-logit(ps));idx={c:np.flatnonzero(y==c) for c in [0,1]};draw=np.column_stack([rng.choice(idx[c],size=(5000,len(idx[c])),replace=True) for c in [0,1]])
        for model in ['spectral_covariance_selector','entropy_hjorth']:
            other=sub[sub.model==model].set_index('participant_id').loc[base.index];assert np.array_equal(y,other.y.to_numpy())
            for correction in ['original','oracle_prior_odds']:
                pb=base.p.to_numpy();pe=other.p.to_numpy()
                if correction!='original':pb=expit(logit(np.clip(pb,1e-8,1-1e-8))+offset);pe=expit(logit(np.clip(pe,1e-8,1-1e-8))+offset)
                mb=metrics(y,pb);me=metrics(y,pe);d=loss(y,pb)-loss(y,pe);ds=d[draw].mean(axis=1);eq=.5*(d[draw[:,:len(idx[0])]].mean(axis=1)+d[draw[:,len(idx[0]):]].mean(axis=1));q=np.quantile(ds,[.025,.975]);qe=np.quantile(eq,[.025,.975])
                contrasts.append(dict(direction=direction,model=model,correction=correction,source_fraction=ps,target_fraction=pt,offset=offset,baseline_logloss=mb['logloss'],augmented_logloss=me['logloss'],delta_ll=float(d.mean()),equal_class_delta_ll=float((d[y==0].mean()+d[y==1].mean())/2),control_delta_ll=float(d[y==0].mean()),adhd_delta_ll=float(d[y==1].mean()),lower=q[0],upper=q[1],equal_lower=qe[0],equal_upper=qe[1],delta_auroc=me['auroc']-mb['auroc'],delta_brier=mb['brier']-me['brier'],mean_baseline_p=pb.mean(),mean_augmented_p=pe.mean(),baseline_auroc=mb['auroc'],augmented_auroc=me['auroc'],baseline_brier=mb['brier'],augmented_brier=me['brier']))
                for pid,yy,p0,p1 in zip(base.index,y,pb,pe):records.append(dict(direction=direction,model=model,correction=correction,participant_id=pid,y=yy,p_baseline=p0,p_augmented=p1))
                for i,(v,e) in enumerate(zip(ds,eq)):drawrows.append(dict(direction=direction,model=model,correction=correction,draw=i,delta_ll=v,equal_class_delta_ll=e))
    csv(out/'PREDICTIONS.csv',records);csv(out/'CONTRASTS.csv',contrasts);csv(out/'CONDITIONAL_DRAWS.csv',drawrows);dump(out/'COMPLETE.json',dict(status='PASS',predictive_refits=0,comparisons=4,source_prediction_sha256=sha(p),target_label_oracle=True))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('part',choices=['extract','age','bandwidth','offset']);ap.add_argument('--release',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--raw',type=Path);ap.add_argument('--transfer',type=Path);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    plan=args.output/'ANALYSIS_PLAN.txt';lock=json.loads((args.output/'audit/PLAN_LOCK.json').read_text());assert sha(plan)==lock['plan_sha256']
    out=args.output/args.part;out.mkdir(exist_ok=False);init(args.release,args.output);data=CORE.load_data()
    assert len(data['ids'])==96 and data['y'].sum()==41
    dump(out/'RUN.json',dict(start_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=sha(plan),code_sha256=sha(__file__),python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,participants=data['ids'],workers=args.workers))
    tests(out)
    try:
        if args.part=='extract':extract(args,data,out)
        elif args.part=='offset':offset(args,data,out)
        else:refit(args,data,out)
    except Exception as e:dump(out/'FAILURE.json',dict(error=repr(e),utc=datetime.now(timezone.utc).isoformat()));raise
if __name__=='__main__':main()
