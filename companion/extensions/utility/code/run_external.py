"""Post-hoc conceptual replication on openly released independent adult data."""
from pathlib import Path
import json,hashlib,time
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import run_extension as core
ROOT=core.ROOT;SRC=ROOT/'external';OUT=ROOT/'results/external'
def transform(a,b):
 keep=~np.isnan(a).all(axis=0)
 imputer=SimpleImputer(strategy='median');aa=imputer.fit_transform(a[:,keep]);sc=StandardScaler().fit(aa)
 return sc.transform(aa),sc.transform(imputer.transform(b[:,keep])),keep
def run(release_root=None):
 lock=json.loads((ROOT/'audit/EXTERNAL_PLAN_LOCK.json').read_text());assert core.sha(ROOT/'EXTERNAL_ANALYSIS_PLAN.txt')==lock['sha256']
 if OUT.exists() and any(OUT.iterdir()):raise RuntimeError('Existing output retained; no overwrite')
 OUT.mkdir(parents=True,exist_ok=True)
 manifest=json.loads((SRC/'DOWNLOAD_MANIFEST.json').read_text())
 assert all(core.sha(SRC/f['name'])==f['sha256'] for f in manifest)
 release=release_root or json.loads((ROOT/'audit/INPUT_MANIFEST.json').read_text())['release_root'];core.load_frozen(release)
 train=pd.read_csv(SRC/'MLBIAS_trainingSet_open.csv',sep=';',decimal=',');test=pd.read_csv(SRC/'MLBIAS_testSet_open.csv',sep=';',decimal=',')
 assert len(train)==50 and len(test)==36 and train.columns.tolist()==test.columns.tolist()
 y=train.Group.map({'HC':0,'ADHD_NoMed':1});yt=test.Group.map({'HC':0,'ADHD':1});assert y.notna().all() and yt.notna().all();y=y.to_numpy(int);yt=yt.to_numpy(int)
 columns=train.columns.drop('Group');train[columns]=train[columns].apply(pd.to_numeric,errors='raise');test[columns]=test[columns].apply(pd.to_numeric,errors='raise')
 assert not np.isinf(train[columns].to_numpy()).any() and not np.isinf(test[columns].to_numpy()).any()
 eeg=[c for c in columns if c.startswith('EEG_')];taskeye=[c for c in columns if c.startswith(('CPT_','ET_'))];non_eeg=[c for c in columns if c not in eeg]
 assert (len(taskeye),len(eeg),len(non_eeg))==(49,18,58)
 missing=[{'split':split,'feature':c,'missing_n':int(df[c].isna().sum()),'n':len(df)} for split,df in [('train',train),('test',test)] for c in columns]
 core.save(OUT/'MISSINGNESS.csv',missing)
 folds=list(StratifiedKFold(5,shuffle=True,random_state=20261131).split(train,y))
 duplicates=[]
 for i,a in enumerate(train[columns].to_numpy()):
  for j,b in enumerate(test[columns].to_numpy()):
   if np.array_equal(a,b,equal_nan=True):duplicates.append([i,j])
 core.dump(OUT/'DATA_AUDIT.json',{'train_n':len(y),'test_n':len(yt),'training_class_counts':np.bincount(y).tolist(),'test_class_counts':np.bincount(yt).tolist(),'cross_split_exact_duplicate_predictor_rows':duplicates,'source_identity_check':'source row numbers only; published independent recruitment, no independent identity verification','input_sha256':{f['name']:f['sha256'] for f in manifest}})
 core.dump(OUT/'SPLIT.json',{'train_rows':list(range(50)),'test_rows':list(range(36)),'inner':[{'train':a.tolist(),'validation':b.tolist()} for a,b in folds]})
 predictions=[];tuning=[];foldscores=[];dropped=[];metrics=[];replay=[];fits=0
 for baseline,basecols in [('task_eye',taskeye),('all_non_eeg',non_eeg)]:
  for model,cols in [('baseline',basecols),('eeg_augmented',basecols+eeg)]:
   x=train[cols].to_numpy(float);v=test[cols].to_numpy(float);mats=[]
   for k,(a,b) in enumerate(folds):
    aa,bb,keep=transform(x[a],x[b]);mats.append((aa,bb));dropped.append({'baseline':baseline,'model':model,'partition':k,'dropped_columns':[c for c,z in zip(cols,keep) if not z]})
   xx,vv,keep=transform(x,v);probe,_,_=transform(x,v+17);assert np.array_equal(xx,probe)
   dropped.append({'baseline':baseline,'model':model,'partition':'final','dropped_columns':[c for c,z in zip(cols,keep) if not z]})
   for learner in core.GRIDS:
    ci,scores,fs,retry=core.choose(mats,[y[a] for a,b in folds],[y[b] for a,b in folds],learner,[20261133+j for j in range(5)])
    fits+=5*len(core.GRIDS[learner]);fitted,retry2=core.fit(xx,y,learner,core.GRIDS[learner][ci],20261232);fits+=1
    pp=fitted.predict_proba(vv)[:,1];assert np.isfinite(pp).all()
    for j,score in enumerate(scores):tuning.append({'baseline':baseline,'model':model,'learner':learner,'candidate':j,'parameters':json.dumps(core.GRIDS[learner][j],sort_keys=True),'inner_logloss':score,'selected':j==ci})
    for f in fs:foldscores.append({'baseline':baseline,'model':model,'learner':learner,**f})
    for i,(label,p) in enumerate(zip(yt,pp)):predictions.append({'baseline':baseline,'model':model,'learner':learner,'source_test_row':i,'y':int(label),'p':float(p)})
    metrics.append({'baseline':baseline,'model':model,'learner':learner,'predictors':len(cols),'test_n':len(yt),**core.metrics(yt,pp)})
    fitted2,_=core.fit(xx,y,learner,core.GRIDS[learner][ci],20261232);p2=fitted2.predict_proba(vv)[:,1];assert np.array_equal(pp,p2)
    replay.append({'baseline':baseline,'model':model,'learner':learner,'max_abs_difference':float(np.max(abs(pp-p2)))})
 core.save(OUT/'PREDICTIONS.csv',predictions);core.save(OUT/'TUNING.csv',tuning);core.save(OUT/'INNER_FOLD_SCORES.csv',foldscores);core.save(OUT/'METRICS.csv',metrics);core.dump(OUT/'DROPPED_COLUMNS.json',dropped)
 pred=pd.DataFrame(predictions);rng=np.random.default_rng(20261137);ix=np.column_stack([rng.choice(np.flatnonzero(yt==c),size=(10000,sum(yt==c)),replace=True) for c in [0,1]])
 boot=[];increments=[];participant=[]
 for (baseline,learner),d in pred.groupby(['baseline','learner']):
  p0=d[d.model=='baseline'].sort_values('source_test_row').p.to_numpy();p1=d[d.model=='eeg_augmented'].sort_values('source_test_row').p.to_numpy()
  dl=core.loss(yt,p0)-core.loss(yt,p1);db=(yt-p0)**2-(yt-p1)**2
  points={'delta_logloss':float(dl.mean()),'delta_auroc':float(core.roc_auc_score(yt,p1)-core.roc_auc_score(yt,p0)),'delta_brier':float(db.mean())}
  vals={'delta_logloss':dl[ix].mean(axis=1),'delta_brier':db[ix].mean(axis=1)}
  # Rank-equivalent pair comparison is efficient because resamples preserve both classes.
  n0=sum(yt==0);p1b=p1[ix];p0b=p0[ix]
  def auc(p):
   diff=p[:,n0:,None]-p[:,None,:n0];return ((diff>0)+.5*(diff==0)).mean(axis=(1,2))
  vals['delta_auroc']=auc(p1b)-auc(p0b)
  row={'baseline':baseline,'learner':learner,**points,'group0_delta_logloss':float(dl[yt==0].mean()),'group1_delta_logloss':float(dl[yt==1].mean()),'participants_positive':int(sum(dl>0))}
  for k,v in vals.items():row[k+'_conditional_low'],row[k+'_conditional_high']=np.quantile(v,[.025,.975]).tolist()
  increments.append(row)
  for b in range(10000):boot.append({'baseline':baseline,'learner':learner,'bootstrap':b,**{k:float(v[b]) for k,v in vals.items()}})
  for i in range(36):participant.append({'baseline':baseline,'learner':learner,'source_test_row':i,'y':int(yt[i]),'delta_logloss':float(dl[i]),'delta_brier':float(db[i])})
 core.save(OUT/'INCREMENTS.csv',increments);core.save(OUT/'CONDITIONAL_RESAMPLES.csv',boot);core.save(OUT/'PARTICIPANT_INCREMENTS.csv',participant)
 core.dump(OUT/'REPLAY.json',replay)
 assert all(core.sha(SRC/f['name'])==f['sha256'] for f in manifest)
 core.dump(OUT/'DONE.json',{'planned_fit_calls':fits,'verification_fit_calls':12,'expected_planned_fit_calls':292,'heldout_transform_invariance':True,'inputs_unchanged':True,'plan_sha256':lock['sha256'],'code_sha256':core.sha(__file__)})
 print(pd.DataFrame(increments).to_string(index=False),flush=True)
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--release-root');run(parser.parse_args().release_root)
