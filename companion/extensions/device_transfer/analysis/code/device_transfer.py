"""Exploratory device-transfer stress test; plan locked before new results.

Uses the immutable release's training-only feature builders. No raw extraction,
target adaptation or changes to original results. Run with Python -B.
"""
from pathlib import Path
import os,sys,json,hashlib,importlib.util,platform
sys.dont_write_bytecode=True
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2]
PROJECT=ROOT.parents[1]
REPO=PROJECT/'eeg-gain-github'
OUT=ROOT/'analysis/results'
if (OUT/'COMPLETION.json').exists():raise RuntimeError('Completed results exist; use a separate output directory for any documented replay.')
OUT.mkdir(parents=True,exist_ok=True)
os.environ['EEG_RELEASE_ROOT']=str(REPO)
os.environ['EEG_RUN_ROOT']=str(ROOT/'analysis/runtime')
import numpy as np,pandas as pd,scipy,sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
threadpool_limits(1)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,x): (OUT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def save(name,rows):pd.DataFrame(rows).to_csv(OUT/name,index=False,float_format='%.17g')
lock=json.loads((ROOT/'audit/PLAN_LOCK.json').read_text())
assert sha(ROOT/'ANALYSIS_PLAN.txt')==lock['sha256']
paths=[p for p in REPO.rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts]
input_hashes={str(p.relative_to(REPO)):sha(p) for p in paths}
dump('INPUT_MANIFEST.json',{'repository':str(REPO),'files':input_hashes,'plan_sha256':lock['sha256'],'analysis_code_sha256':sha(Path(__file__))})
spec=importlib.util.spec_from_file_location('p3_posthoc',REPO/'reproduce/run_posthoc.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
data=mod.load_data()
ids=np.asarray(data['ids']);y=data['y'];device=data['qc'].device.to_numpy()
assert len(ids)==96 and len(set(ids))==96 and y.sum()==41
assert set(device)=={'Epoc X','Epoc+'}
kinds={'baseline':'none','regional_spectral':'E0','expanded_spectral':'E1','covariance':'E2','entropy_hjorth':'E4'}
ordered=list(kinds)
fold_rows=[];scores=[];predictions=[];fit_rows=[];coefs=[];split_details=[]
fit_count=0;retries=0
for direction,source,target,seed in [('X_to_Plus','Epoc X','Epoc+',20261101),('Plus_to_X','Epoc+','Epoc X',20261102)]:
 tr=np.flatnonzero(device==source);te=np.flatnonzero(device==target)
 assert not set(tr)&set(te) and len(tr)+len(te)==96
 assert min(np.bincount(y[tr],minlength=2))>=5
 split_details.append({'direction':direction,'source_device':source,'target_device':target,'source_n':len(tr),'target_n':len(te),'source_adhd':int(y[tr].sum()),'target_adhd':int(y[te].sum()),'inner_seed':seed})
 inner=[]
 for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(tr,y[tr])):
  ia,ib=tr[a],tr[b]
  assert not set(ia)&set(ib) and not (set(ia)|set(ib))&set(te)
  inner.append((ia,ib,{}))
  for role,indices in [('train',ia),('validation',ib)]:
   for idx in indices:fold_rows.append({'direction':direction,'level':'inner','fold':fold,'role':role,'participant_id':ids[idx]})
 for role,indices in [('source',tr),('target',te)]:
  for idx in indices:fold_rows.append({'direction':direction,'level':'outer','fold':-1,'role':role,'participant_id':ids[idx]})
 bests={};final_cache={};fitted={}
 for name,kind in kinds.items():
  matrices=[mod.design(data,ia,ib,kind,0,cache) for ia,ib,cache in inner]
  per_c={}
  for c in mod.GRID:
   all_loss=[]
   for (ia,ib,_),(a,b,_) in zip(inner,matrices):
    model,retry=mod.fit(a,y[ia],c);fit_count+=1;retries+=int(retry)
    all_loss.extend(mod.loss(y[ib],model.predict_proba(b)[:,1]).tolist())
   per_c[c]=float(np.mean(all_loss))
  best=min(mod.GRID,key=lambda c:(round(per_c[c],12),c));bests[name]=(best,per_c[best])
  for c,v in per_c.items():scores.append({'direction':direction,'model':name,'C':c,'pooled_inner_logloss':v,'selected_within_model':c==best})
  a,b,names=mod.design(data,tr,te,kind,0,final_cache)
  # Perturb the assessment rows only: source transforms must remain unchanged.
  probe={**data,'raw':{k:v.copy() for k,v in data['raw'].items()},'e4':data['e4'].copy()}
  for key,v in probe['raw'].items():
   if key!='covariances':v[te]=v[te]+17.0
   else:
    for idx in te:
     v[idx]=np.eye(v.shape[-1])[None,:,:]/v.shape[-1]
  probe['e4'][te]+=17.0
  pa,_,_=mod.design(probe,tr,te,kind,0,{})
  assert np.allclose(a,pa,rtol=0,atol=1e-12),(direction,name,'assessment leakage')
  model,retry=mod.fit(a,y[tr],best);fit_count+=1;retries+=int(retry)
  p=model.predict_proba(b)[:,1]
  assert np.isfinite(p).all() and ((p>0)&(p<1)).all()
  fitted[name]=p
  for n,c in zip(names,model.coef_[0]):coefs.append({'direction':direction,'model':name,'feature':n,'coefficient':float(c),'intercept':float(model.intercept_[0]),'C':best})
  fit_rows.append({'direction':direction,'model':name,'C':best,'inner_logloss':per_c[best],'source_n':len(tr),'target_n':len(te),'selected_representation':name})
 selector=min(['regional_spectral','expanded_spectral','covariance'],key=lambda n:(round(bests[n][1],12),ordered.index(n),bests[n][0]))
 fitted['spectral_covariance_selector']=fitted[selector].copy()
 fit_rows.append({'direction':direction,'model':'spectral_covariance_selector','C':bests[selector][0],'inner_logloss':bests[selector][1],'source_n':len(tr),'target_n':len(te),'selected_representation':selector})
 for name,p in fitted.items():
  for idx,value in zip(te,p):predictions.append({'direction':direction,'model':name,'participant_id':ids[idx],'source_device':source,'target_device':target,'y':int(y[idx]),'p':float(value)})
 print(direction+': source-only tuning and final fits completed',flush=True)
assert fit_count==260,fit_count
save('FOLD_MANIFEST.csv',fold_rows);save('INNER_SCORES.csv',scores);save('MODEL_SELECTION.csv',fit_rows);save('PREDICTIONS.csv',predictions);save('COEFFICIENTS.csv',coefs)
dump('SPLIT_DETAILS.json',split_details)
pred=pd.DataFrame(predictions)
metric_rows=[];contrasts=[];classrows=[];draws=[]
rng=np.random.default_rng(20261103)
for direction,frame in pred.groupby('direction',sort=False):
 base=frame[frame.model=='baseline'].set_index('participant_id').sort_index()
 yy=base.y.to_numpy(int);p0=base.p.to_numpy(float);n=len(base)
 group_indices=[np.flatnonzero(yy==j) for j in [0,1]]
 draw_idx=np.column_stack([rng.choice(ind,(5000,len(ind)),replace=True) for ind in group_indices])
 for name,sub in frame.groupby('model',sort=False):
  pp=sub.set_index('participant_id').loc[base.index,'p'].to_numpy(float)
  metrics=mod.metrics(yy,pp)
  metric_rows.append({'direction':direction,'model':name,'n':n,**metrics})
  if name=='baseline':continue
  delta=mod.loss(yy,p0)-mod.loss(yy,pp)
  point={'direction':direction,'model':name,'n':n,'delta_ll':float(delta.mean()),'delta_auroc':float(roc_auc_score(yy,pp)-roc_auc_score(yy,p0)),'delta_brier':float(np.mean((yy-p0)**2-(yy-pp)**2)),'equal_class_delta_ll':float(np.mean([delta[yy==j].mean() for j in [0,1]])),'favorable_participants':int((delta>0).sum())}
  for j in [0,1]:classrows.append({'direction':direction,'model':name,'group':j,'n':int((yy==j).sum()),'delta_ll':float(delta[yy==j].mean())})
  if name in ['spectral_covariance_selector','entropy_hjorth']:
   ll=delta[draw_idx].mean(axis=1)
   br=(((yy-p0)**2-(yy-pp)**2)[draw_idx]).mean(axis=1)
   auc=np.asarray([roc_auc_score(yy[ix],pp[ix])-roc_auc_score(yy[ix],p0[ix]) for ix in draw_idx])
   bal=np.asarray([np.mean([delta[ix][yy[ix]==j].mean() for j in [0,1]]) for ix in draw_idx])
   for metric,values in [('delta_ll',ll),('delta_brier',br),('delta_auroc',auc),('equal_class_delta_ll',bal)]:
    low,high=np.quantile(values,[.025,.975],method='linear');point[metric+'_conditional_low']=float(low);point[metric+'_conditional_high']=float(high)
   for b in range(5000):draws.append({'direction':direction,'model':name,'draw':b,'delta_ll':float(ll[b]),'delta_brier':float(br[b]),'delta_auroc':float(auc[b]),'equal_class_delta_ll':float(bal[b])})
  contrasts.append(point)
save('ABSOLUTE_METRICS.csv',metric_rows);save('CONTRASTS_ALL.csv',contrasts);save('GROUP_CONTRASTS.csv',classrows);save('CONDITIONAL_DRAWS.csv',draws)
principal=[r for r in contrasts if r['model'] in ['spectral_covariance_selector','entropy_hjorth']]
save('PRINCIPAL_CONTRASTS.csv',principal)
assert all(sha(REPO/k)==v for k,v in input_hashes.items())
dump('COMPLETION.json',{'plan_sha256':lock['sha256'],'analysis_code_sha256':sha(Path(__file__)),'software':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},'inner_fits':250,'final_fits':10,'convergence_retries':retries,'source_files_unchanged':len(input_hashes),'source_target_disjoint':True,'training_transforms_invariant_to_assessment_perturbation':True,'conditional_bootstrap_draws_per_contrast':5000,'status':'exploratory; no target adaptation; conditional ranges omit source-training uncertainty','all_principal_contrasts_preserved':len(principal)})
print(pd.DataFrame(principal)[['direction','model','delta_ll','delta_auroc','delta_brier','equal_class_delta_ll']].to_string(index=False),flush=True)
