"""Bounded post-hoc extension. Frozen inputs are read-only; completed jobs are never overwritten."""
from pathlib import Path
import os,sys,json,hashlib,importlib.util,argparse,warnings,platform,time
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import numpy as np,pandas as pd,scipy,sklearn
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from concurrent.futures import ProcessPoolExecutor,as_completed
from threadpoolctl import threadpool_limits
threadpool_limits(1)
ROOT=Path(__file__).resolve().parents[1]
MOD=None;DATA=None;RELEASE=None
KINDS={'baseline':'none','regional_spectral':'E0','expanded_spectral':'E1','covariance':'E2','entropy_hjorth':'E4'}
GRIDS={'logistic':[{'C':c} for c in [.01,.1,1.,10.,100.]],
       'rbf_svm':[{'C':c,'gamma_factor':g} for c in [.1,1.,10.] for g in [.1,1.]],
       'forest':[{'max_depth':3,'min_samples_leaf':5},{'max_depth':None,'min_samples_leaf':5},{'max_depth':None,'min_samples_leaf':10}]}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def save(p,rows):pd.DataFrame(rows).to_csv(p,index=False,float_format='%.17g')
def load_frozen(release):
 global MOD,RELEASE
 RELEASE=Path(release);os.environ['EEG_RELEASE_ROOT']=str(RELEASE);os.environ['EEG_RUN_ROOT']=str(ROOT/'runtime')
 spec=importlib.util.spec_from_file_location('frozen_posthoc',RELEASE/'reproduce/run_posthoc.py')
 MOD=importlib.util.module_from_spec(spec);spec.loader.exec_module(MOD)
def initialize(release,data=None):
 global DATA
 threadpool_limits(1);load_frozen(release);DATA=data
def loss(y,p):
 p=np.clip(p,1e-8,1-1e-8);return -y*np.log(p)-(1-y)*np.log1p(-p)
def metrics(y,p):return {'logloss':float(loss(y,p).mean()),'auroc':float(roc_auc_score(y,p)),'brier':float(np.mean((y-p)**2))}
def fit(x,y,learner,params,seed):
 if learner=='logistic':return MOD.fit(x,y,params['C'])
 if learner=='rbf_svm':
  model=SVC(C=params['C'],gamma=params['gamma_factor']/x.shape[1],kernel='rbf',probability=True,random_state=int(seed),shrinking=True,tol=1e-3,cache_size=128)
 else:model=RandomForestClassifier(n_estimators=128,max_features=1.0,bootstrap=True,class_weight=None,criterion='gini',n_jobs=1,random_state=int(seed),**params)
 with warnings.catch_warnings(record=True) as caught:
  warnings.simplefilter('always');model.fit(x,y)
  relevant=[str(w.message) for w in caught if 'deprecated' not in str(w.message).lower()]
  if relevant:raise RuntimeError((learner,relevant))
 return model,False
def choose(mats,ytr,yvalid,learner,seeds):
 scores=[];fold_scores=[];retries=0
 for ci,params in enumerate(GRIDS[learner]):
  ls=[]
  for j,((a,b),ya,yb,seed) in enumerate(zip(mats,ytr,yvalid,seeds)):
   model,retry=fit(a,ya,learner,params,seed);retries+=int(retry);p=model.predict_proba(b)[:,1]
   assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
   ll=loss(yb,p);ls.extend(ll.tolist());fold_scores.append({'candidate':ci,'inner_fold':j,'n':len(yb),'logloss':float(ll.mean())})
  scores.append(float(np.mean(ls)))
 best=min(range(len(scores)),key=lambda i:(round(scores[i],12),i))
 return best,scores,fold_scores,retries
def empirical_job(spec):
 start=time.monotonic();rep,fold,fraction,width=spec
 name=f'r{rep:02}_f{fold}_n{int(fraction*100):03}_q{width}'
 out=ROOT/'results/empirical_v2'/name
 if (out/'DONE.json').exists():return {'job':name,'existing':True}
 if out.exists() and any(out.iterdir()):raise RuntimeError('Partial job preserved; diagnose before retry: '+str(out))
 out.mkdir(parents=True,exist_ok=True)
 data=DATA;outer=next(o for o in data['folds']['outer'] if o['repeat']==rep and o['outer_fold']==fold)
 y=data['y'];ids=np.asarray(data['ids']);tr=np.asarray(outer['outer_train']);te=np.asarray(outer['outer_test'])
 if fraction<1:
  rng=np.random.default_rng(20261107+rep*100+fold);parts=[]
  for cls in [0,1]:
   members=tr[y[tr]==cls].copy();rng.shuffle(members);parts.extend(members[:int(np.floor(fraction*len(members)))])
  tr=np.sort(parts);fi=[.5,.75].index(fraction)
  inner=[(tr[a],tr[b]) for a,b in StratifiedKFold(5,shuffle=True,random_state=20261109+rep*1000+fold*10+fi).split(tr,y[tr])]
 else:inner=[(np.asarray(i['inner_train']),np.asarray(i['inner_valid'])) for i in outer['inner_folds']]
 assert not set(tr)&set(te) and min(np.bincount(y[tr]))>=5
 for a,b in inner:assert not set(a)&set(b) and not (set(a)|set(b))&set(te) and set(a)|set(b)==set(tr)
 learners=['logistic','rbf_svm','forest'] if fraction==1 and width==0 else (['rbf_svm','forest'] if width else ['logistic','rbf_svm'])
 cache=[{} for _ in inner];ocache={};pred=[];selections=[];foldscores=[];choices=[];fits=0;retries=0
 details={'repeat':rep,'outer_fold':fold,'fraction':fraction,'adjustment':width,'train_n':len(tr),'test_n':len(te)}
 dump(out/'SPLIT.json',{**details,'train_ids':ids[tr].tolist(),'test_ids':ids[te].tolist(),'inner':[{'train_ids':ids[a].tolist(),'validation_ids':ids[b].tolist()} for a,b in inner]})
 best_by={l:{} for l in learners};prob_by={l:{} for l in learners}
 for model,kind in KINDS.items():
  mats=[MOD.design(data,a,b,kind,width,c)[:2] for (a,b),c in zip(inner,cache)]
  x,v,_=MOD.design(data,tr,te,kind,width,ocache)
  # One held-out perturbation check for each representation/design per task.
  probe={**data,'raw':{k:z.copy() for k,z in data['raw'].items()},'e4':data['e4'].copy(),'nuisance':data['nuisance'].copy()}
  for k,z in probe['raw'].items():
   if k=='covariances':z[te]=np.eye(z.shape[-1])[None,None,:,:]/z.shape[-1]
   else:z[te]+=17
  probe['e4'][te]+=17;probe['nuisance'][te]+=17
  px,_,_=MOD.design(probe,tr,te,kind,width,{})
  assert np.allclose(x,px,rtol=0,atol=1e-12)
  for learner in learners:
   seeds=[20261101+rep*10000+fold*100+j+1 for j in range(5)]
   ci,scores,fs,retry=choose(mats,[y[a] for a,b in inner],[y[b] for a,b in inner],learner,seeds)
   fits+=5*len(GRIDS[learner]);retries+=retry
   for j,score in enumerate(scores):selections.append({**details,'learner':learner,'model':model,'candidate':j,'parameters':json.dumps(GRIDS[learner][j],sort_keys=True),'inner_logloss':score,'selected':j==ci})
   for f in fs:foldscores.append({**details,'learner':learner,'model':model,**f})
   fitted,retry=fit(x,y[tr],learner,GRIDS[learner][ci],20261101+rep*10000+fold*100+100);fits+=1;retries+=int(retry)
   pp=fitted.predict_proba(v)[:,1];assert np.isfinite(pp).all() and ((pp>=0)&(pp<=1)).all()
   best_by[learner][model]=(ci,scores[ci]);prob_by[learner][model]=pp
   for idx,value in zip(te,pp):pred.append({**details,'learner':learner,'model':model,'participant_id':ids[idx],'y':int(y[idx]),'p':float(value)})
 for learner in learners:
  order=['regional_spectral','expanded_spectral','covariance']
  selected=min(order,key=lambda k:(round(best_by[learner][k][1],12),order.index(k),best_by[learner][k][0]))
  choices.append({**details,'learner':learner,'selected_representation':selected,'candidate':best_by[learner][selected][0]})
  for idx,value in zip(te,prob_by[learner][selected]):pred.append({**details,'learner':learner,'model':'spectral_covariance_selector','participant_id':ids[idx],'y':int(y[idx]),'p':float(value)})
 if fraction==1 and width==0:
  # All three prespecified/reference outputs checked in each outer split.
  reference=data['old'][(data['old'].repeat==rep)&(data['old'].outer_fold==fold)].set_index('participant_id').loc[ids[te]]
  for model,col in [('baseline','p_m2'),('spectral_covariance_selector','p_adaptive_m3')]:
   current=np.asarray([r['p'] for r in pred if r['learner']=='logistic' and r['model']==model]);assert np.max(np.abs(current-reference[col].to_numpy()))<1e-7,(name,model)
  eold=pd.read_csv(RELEASE/'audit/exploratory_eeg_battery/OUTER_PREDICTIONS.csv');eold=eold[(eold.family=='E4')&(eold.repeat==rep)&(eold.outer_fold==fold)].set_index('participant_id').loc[ids[te]]
  assert np.max(np.abs(prob_by['logistic']['entropy_hjorth']-eold.p_eeg.to_numpy()))<1e-7
 save(out/'PREDICTIONS.csv',pred);save(out/'TUNING.csv',selections);save(out/'INNER_FOLD_SCORES.csv',foldscores);save(out/'SELECTOR_CHOICES.csv',choices)
 result={'job':name,**details,'fit_calls':fits,'convergence_retries':retries,'heldout_perturbation_invariance':True,'seconds':time.monotonic()-start}
 dump(out/'DONE.json',result);return result

SCENARIOS=['redundant','complementary_linear','complementary_nonlinear','acquisition_association']
def generated(scenario,rep,n,which):
 rng=np.random.default_rng(np.random.SeedSequence([20261111,SCENARIOS.index(scenario),rep,which]))
 B=rng.normal(size=(n,7));E=rng.normal(size=(n,9));eta=-.3+.8*B[:,0]-.6*B[:,1]+.4*B[:,2]
 if scenario=='complementary_linear':p=expit(eta+.8*E[:,0])
 elif scenario=='complementary_nonlinear':p=expit(eta+.9*(E[:,0]**2-1)/np.sqrt(2))
 else:p=expit(eta)
 y=(rng.random(n)<p).astype(int);u=rng.random(n)
 def observed(shift):
  e=E.copy();D=(u<(.5 if shift else np.where(y==1,.8,.2))).astype(int)
  if scenario=='acquisition_association':e[:,:3]+=(2*D-1)[:,None]
  Z=.7*B[:,np.arange(9)%7]+np.sqrt(.51)*e
  return np.column_stack([B,Z]),D
 X,D=observed(False);shiftX,shiftD=observed(True)
 return {'B':B,'E':E,'eta':eta,'generating_p':p,'y':y,'X':X,'D':D,'shiftX':shiftX,'shiftD':shiftD}
def oracle(scenario,d,shift=False,nodes=80):
 eta=d['eta'];p=expit(eta)
 if scenario.startswith('complementary'):
  q,w=np.polynomial.hermite.hermgauss(nodes);e=np.sqrt(2)*q
  extra=.8*e if scenario.endswith('linear') and not scenario.endswith('nonlinear') else .9*(e**2-1)/np.sqrt(2)
  pbase=expit(eta[:,None]+extra[None,:])@(w/np.sqrt(np.pi));padd=d['generating_p']
 elif scenario=='acquisition_association' and not shift:
  X=d['X'];res=(X[:,7:10]-.7*d['B'][:,:3])/np.sqrt(.51)
  # Gaussian likelihood ratio g_plus/g_minus = exp(2*sum(res)). Stable mixture log-ratio.
  lr=2*res.sum(axis=1)
  logratio=np.logaddexp(np.log(.8)+lr,np.log(.2))-np.logaddexp(np.log(.2)+lr,np.log(.8))
  pbase=p;padd=expit(eta+logratio)
 else:pbase=p;padd=p
 pbase=np.clip(pbase,1e-12,1-1e-12);padd=np.clip(padd,1e-12,1-1e-12)
 kl=padd*np.log(padd/pbase)+(1-padd)*np.log((1-padd)/(1-pbase))
 assert np.min(kl)>-1e-12
 return pbase,padd,float(kl.mean())
def simulation_job(spec):
 start=time.monotonic();scenario,rep,n=spec;name=f'{scenario}_r{rep:02}_n{n}'
 out=ROOT/'results/simulation'/name
 if (out/'DONE.json').exists():return {'job':name,'existing':True}
 if out.exists() and any(out.iterdir()):raise RuntimeError('Partial output preserved: '+str(out))
 out.mkdir(parents=True,exist_ok=True)
 train=generated(scenario,rep,384,0);test=generated(scenario,rep,4096,1)
 X=train['X'][:n];y=train['y'][:n];assert min(np.bincount(y,minlength=2))>=5
 folds=list(StratifiedKFold(5,shuffle=True,random_state=20261113+rep).split(X,y))
 tuning=[];foldrows=[];metricrows=[];pred={};fits=0;retries=0
 assessment=['iid','shifted'] if scenario=='acquisition_association' else ['iid']
 for model,cols in [('baseline',7),('eeg_augmented',16)]:
  xx=X[:,:cols];mats=[]
  for a,b in folds:
   sc=StandardScaler().fit(xx[a]);mats.append((sc.transform(xx[a]),sc.transform(xx[b])))
  sc=StandardScaler().fit(xx);xf=sc.transform(xx)
  for learner in ['logistic','rbf_svm']:
   seeds=[20261121+rep*10000+n*10+j for j in range(5)]
   ci,scores,fs,retry=choose(mats,[y[a] for a,b in folds],[y[b] for a,b in folds],learner,seeds);fits+=5*len(GRIDS[learner]);retries+=retry
   for j,v in enumerate(scores):tuning.append({'learner':learner,'model':model,'candidate':j,'parameters':json.dumps(GRIDS[learner][j],sort_keys=True),'inner_logloss':v,'selected':j==ci})
   for f in fs:foldrows.append({'learner':learner,'model':model,**f})
   fitted,retry=fit(xf,y,learner,GRIDS[learner][ci],20261121+rep*10000+n*10+99);fits+=1;retries+=int(retry)
   for condition in assessment:
    xt=test['X'] if condition=='iid' else test['shiftX'];pp=fitted.predict_proba(sc.transform(xt[:,:cols]))[:,1]
    assert np.isfinite(pp).all() and ((pp>=0)&(pp<=1)).all()
    pred[condition+'__'+learner+'__'+model]=pp
    metricrows.append({'scenario':scenario,'replicate':rep,'train_n':n,'assessment':condition,'learner':learner,'model':model,'test_n':4096,**metrics(test['y'],pp)})
 oracle_rows=[]
 for condition in assessment:
  p0,p1,kl=oracle(scenario,test,condition=='shifted');pred[condition+'__oracle_baseline']=p0;pred[condition+'__oracle_augmented']=p1
  oracle_rows.append({'scenario':scenario,'replicate':rep,'train_n':n,'assessment':condition,'conditional_information_benchmark':kl})
 pred['test_y']=test['y'];pred['train_y']=y;pred['test_B']=test['B'];pred['test_E']=test['E'];pred['test_D']=test['D'];pred['test_shiftD']=test['shiftD']
 np.savez_compressed(out/'PREDICTIONS.npz',**pred)
 save(out/'TUNING.csv',tuning);save(out/'INNER_FOLD_SCORES.csv',foldrows);save(out/'METRICS.csv',metricrows);save(out/'ORACLE.csv',oracle_rows)
 dump(out/'SPLIT.json',{'training_seed':[20261111,SCENARIOS.index(scenario),rep,0],'test_seed':[20261111,SCENARIOS.index(scenario),rep,1],'train_n':n,'test_n':4096,'inner':[{'train':a.tolist(),'validation':b.tolist()} for a,b in folds]})
 result={'job':name,'fit_calls':fits,'convergence_retries':retries,'seconds':time.monotonic()-start};dump(out/'DONE.json',result);return result
def run(args):
 lock=json.loads((ROOT/'audit/PLAN_LOCK.json').read_text());assert sha(ROOT/'ANALYSIS_PLAN.txt')==lock['plan_sha256']
 manifest=json.loads((ROOT/'audit/INPUT_MANIFEST.json').read_text());release=Path(args.release_root or manifest['release_root'])
 assert all(sha(release/k)==v for k,v in manifest['files'].items())
 load_frozen(release);data=MOD.load_data() if args.part=='empirical' else None
 dump(ROOT/'audit'/f'{args.part}_RUN.json',{'plan_sha256':lock['plan_sha256'],'code_sha256':sha(__file__),'software':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},'workers':args.workers,'grids':GRIDS})
 if args.part=='empirical':
  specs=[(r,f,frac,width) for r in range(10) for f in range(5) for frac,width in [(1.,0),(1.,3),(.5,0),(.75,0)]]
  worker=empirical_job
 else:specs=[(s,r,n) for s in SCENARIOS for r in range(20) for n in [96,192,384]];worker=simulation_job
 if args.first_only:specs=specs[:1]
 start=time.monotonic();done=[]
 with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=(str(release),data)) as ex:
  jobs={ex.submit(worker,s):s for s in specs}
  for i,future in enumerate(as_completed(jobs),1):
   try:result=future.result()
   except Exception as e:
    dump(ROOT/'audit'/f'{args.part}_FAILURE.json',{'spec':jobs[future],'error':repr(e),'completed':done});raise
   done.append(result)
   if i==1 or i%10==0 or i==len(specs):print(f'{args.part}: {i}/{len(specs)} jobs complete; {time.monotonic()-start:.0f} s',flush=True)
 assert all(sha(release/k)==v for k,v in manifest['files'].items())
 dump(ROOT/'audit'/f'{args.part}_COMPLETION.json',{'jobs':len(specs),'seconds':time.monotonic()-start,'input_files_unchanged':len(manifest['files']),'code_sha256':sha(__file__),'first_only':args.first_only})
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('part',choices=['empirical','simulation']);parser.add_argument('--release-root');parser.add_argument('--workers',type=int,default=4);parser.add_argument('--first-only',action='store_true');run(parser.parse_args())
