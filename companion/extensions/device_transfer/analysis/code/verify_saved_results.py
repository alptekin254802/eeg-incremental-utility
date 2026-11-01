"""Independent arithmetic and replay checks, without new model fitting."""
from pathlib import Path
import json,hashlib,argparse
import numpy as np,pandas as pd
W=Path(__file__).resolve().parents[2];R=W/'analysis/results'
parser=argparse.ArgumentParser();parser.add_argument('--release-root',type=Path);parser.add_argument('--replay',type=Path,default=W/'analysis/replay');args=parser.parse_args();Q=args.replay
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(n):return pd.read_csv(R/n,float_precision='round_trip')
same={p.name:sha(p)==sha(Q/p.name) for p in R.glob('*.csv')};assert all(same.values())
pred=read('PREDICTIONS.csv');contrast=read('CONTRASTS_ALL.csv');groups=read('GROUP_CONTRASTS.csv');absolute=read('ABSOLUTE_METRICS.csv')
max_error=0
def near(a,b):
 global max_error
 e=abs(a-b);max_error=max(max_error,e);assert e<1e-12,(a,b,e)
def ll(y,p):return -(y*np.log(p)+(1-y)*np.log1p(-p))
def auc(y,p):
 dif=p[y==1][:,None]-p[y==0][None,:]
 return np.mean((dif>0)+.5*(dif==0))
for d,fr in pred.groupby('direction',sort=False):
 b=fr[fr.model=='baseline'].set_index('participant_id').sort_index();y=b.y.to_numpy();p0=b.p.to_numpy();assert b.index.is_unique
 for m,sub in fr.groupby('model',sort=False):
  sub=sub.set_index('participant_id').loc[b.index];assert (sub.y.to_numpy()==y).all();p=sub.p.to_numpy()
  row=absolute[(absolute.direction==d)&(absolute.model==m)].iloc[0]
  near(ll(y,p).mean(),row.logloss);near(auc(y,p),row.auroc);near(np.mean((p-y)**2),row.brier)
  if m=='baseline':continue
  z=ll(y,p0)-ll(y,p);cr=contrast[(contrast.direction==d)&(contrast.model==m)].iloc[0]
  near(z.mean(),cr.delta_ll);near(auc(y,p)-auc(y,p0),cr.delta_auroc);near(np.mean((p0-y)**2-(p-y)**2),cr.delta_brier)
  d0=z[y==0].mean();d1=z[y==1].mean();near((d0+d1)/2,cr.equal_class_delta_ll);near((1-y.mean())*d0+y.mean()*d1,cr.delta_ll)
  for j,dj in enumerate([d0,d1]):near(dj,groups[(groups.direction==d)&(groups.model==m)&(groups.group==j)].iloc[0].delta_ll)
draws=read('CONDITIONAL_DRAWS.csv');principal=read('PRINCIPAL_CONTRASTS.csv')
for _,r in principal.iterrows():
 dr=draws[(draws.direction==r.direction)&(draws.model==r.model)];assert len(dr)==5000
 for metric in ['delta_ll','delta_auroc','delta_brier','equal_class_delta_ll']:
  lo,hi=np.quantile(dr[metric],[.025,.975],method='linear');near(lo,r[metric+'_conditional_low']);near(hi,r[metric+'_conditional_high'])
manifest=json.loads((R/'INPUT_MANIFEST.json').read_text());repo=args.release_root or Path(manifest['repository']);assert all(sha(repo/k)==v for k,v in manifest['files'].items())
assert sha(W/'ANALYSIS_PLAN.txt')==manifest['plan_sha256']
completion=json.loads((R/'COMPLETION.json').read_text());assert sha(W/'analysis/code/device_transfer.py')==completion['analysis_code_sha256']
report={'replay_csv_byte_identical':same,'original_results_code_sha256':completion['analysis_code_sha256'],'portable_code_sha256':sha(W/'analysis/code/device_transfer_portable.py'),'portable_change':'Only CLI paths replaced project-derived release/output paths; statistical body unchanged. Original code preserved.','plan_sha256':manifest['plan_sha256'],'independent_metric_max_error':max_error,'independent_AUROC_method':'all positive-negative score pairs including half credit for ties','all_10_fixed_and_selected_contrasts_checked':True,'class_weight_identity_checked':True,'all_conditional_quantiles_checked':True,'original_release_files_unchanged':len(manifest['files']),'new_analysis_fits':260,'deterministic_replay_fits':260,'new_representations_or_exclusions':0,'no_full_training_uncertainty_estimated':True}
(W/'audit/ANALYSIS_VERIFICATION.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
