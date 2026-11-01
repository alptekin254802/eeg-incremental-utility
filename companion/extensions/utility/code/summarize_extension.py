"""Derive complete tables and figures from the locked extension outputs."""
from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run_extension as c
R=c.ROOT;OUT=R/'reports';FIG=R/'figures';OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
NAMES={'regional_spectral':'Regional spectral','expanded_spectral':'Expanded spectral','covariance':'Spectral + covariance','entropy_hjorth':'Entropy + Hjorth','spectral_covariance_selector':'Spectral/covariance selector'}
LEARN={'logistic':'Logistic regression','rbf_svm':'RBF SVM','forest':'Random forest'}
COL={'logistic':'#245a81','rbf_svm':'#b15b29','forest':'#42795a'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
def savefig(fig,name):
 fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight');fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)
def independent_metrics(y,p):
 # A second formulation validates the metric implementation on saved predictions.
 p=np.clip(p,1e-8,1-1e-8); ll=-np.log(np.where(y==1,p,1-p)).mean()
 dif=p[y==1,None]-p[y==0][None,:];auc=((dif>0)+.5*(dif==0)).mean()
 return {'logloss':float(ll),'auroc':float(auc),'brier':float(np.square(y-p).mean())}
def empirical():
 jobs=list((R/'results/empirical_v2').glob('*/DONE.json'));assert len(jobs)==200
 preds=pd.concat([pd.read_csv(f.parent/'PREDICTIONS.csv') for f in jobs],ignore_index=True)
 key=['fraction','adjustment','learner','model','repeat'];rows=[]
 for keys,g in preds.groupby(key):
  assert len(g)==96 and g.participant_id.nunique()==96
  y=g.y.to_numpy(int);p=g.p.to_numpy();m=c.metrics(y,p);check=independent_metrics(y,p)
  assert all(abs(m[k]-check[k])<1e-12 for k in m)
  rows.append({**dict(zip(key,keys)),'train_n_min':int(g.train_n.min()),'train_n_max':int(g.train_n.max()),**m})
 met=pd.DataFrame(rows);inc=[]
 for _,row in met[met.model!='baseline'].iterrows():
  b=met[(met.model=='baseline')&(met.fraction==row.fraction)&(met.adjustment==row.adjustment)&(met.learner==row.learner)&(met['repeat']==row['repeat'])].iloc[0]
  inc.append({**row.to_dict(),'baseline_logloss':b.logloss,'baseline_auroc':b.auroc,'baseline_brier':b.brier,'delta_logloss':b.logloss-row.logloss,'delta_auroc':row.auroc-b.auroc,'delta_brier':b.brier-row.brier})
 inc=pd.DataFrame(inc);summ=[]
 for keys,g in inc.groupby(key[:-1]):
  assert len(g)==10
  v=g.delta_logloss.to_numpy();summ.append({**dict(zip(key[:-1],keys)),'train_n_min':int(g.train_n_min.min()),'train_n_max':int(g.train_n_max.max()),**{col:float(g[col].mean()) for col in ['baseline_logloss','logloss','baseline_auroc','auroc','baseline_brier','brier','delta_logloss','delta_auroc','delta_brier']},'delta_logloss_median':float(np.median(v)),'q25':float(np.quantile(v,.25)),'q75':float(np.quantile(v,.75)),'minimum':float(v.min()),'maximum':float(v.max()),'positive_repeats':int(sum(v>0))})
 for name,data in [('EMPIRICAL_PREDICTIONS',preds),('EMPIRICAL_REPEAT_METRICS',met),('EMPIRICAL_REPEAT_INCREMENTS',inc),('EMPIRICAL_SUMMARY',pd.DataFrame(summ))]:data.to_csv(OUT/(name+'.csv'),index=False,float_format='%.17g')
 # Verify all class-stratified smaller training sets are nested and outer tests never enter training.
 for rep in range(10):
  for fold in range(5):
   ss=[json.loads((R/f'results/empirical_v2/r{rep:02}_f{fold}_n{n:03}_q0/SPLIT.json').read_text()) for n in [50,75,100]]
   assert set(ss[0]['train_ids'])<set(ss[1]['train_ids'])<set(ss[2]['train_ids'])
   assert all(set(s['test_ids'])==set(ss[2]['test_ids']) and not set(s['train_ids'])&set(s['test_ids']) for s in ss)
 fig,axes=plt.subplots(1,2,figsize=(8.8,4.2),sharey=True)
 order=list(NAMES)
 for ax,adj in zip(axes,[0,3]):
  ax.axvline(0,color='.55',lw=.8)
  for li,(learner,color) in enumerate(COL.items()):
   for j,model in enumerate(order):
    v=inc[(inc.fraction==1)&(inc.adjustment==adj)&(inc.learner==learner)&(inc.model==model)].delta_logloss.to_numpy()
    if not len(v):continue
    yy=j+(li-1)*.21;ax.scatter(v,np.full(len(v),yy),s=12,color=color,alpha=.35);ax.scatter(v.mean(),yy,s=33,color=color,marker='D',label=LEARN[learner] if j==0 else None)
  ax.set_yticks(range(5),[NAMES[x] for x in order]);ax.set_title('Unadjusted' if adj==0 else 'Device + quality in both models');ax.set_xlabel('Log-loss increment (baseline − EEG)');ax.grid(axis='y',alpha=.12)
  if adj==3:ax.set_xticks([-.10,-.05,0,.05])
  ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
 axes[0].invert_yaxis();h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,frameon=False,loc='upper center',bbox_to_anchor=(.61,1.03),ncol=3,fontsize=9);fig.tight_layout(rect=[0,0,1,.92]);savefig(fig,'model_family_sensitivity')
 fig,axes=plt.subplots(2,2,figsize=(8.8,6.0),sharex=True)
 for col,learner in enumerate(['logistic','rbf_svm']):
  for model,color in [('baseline','#555555'),('spectral_covariance_selector','#245a81'),('entropy_hjorth','#b15b29')]:
   mm=met[(met.adjustment==0)&(met.learner==learner)&(met.model==model)]
   for rep,g in mm.groupby('repeat'):
    g=g.sort_values('fraction');axes[0,col].plot(g.fraction*100,g.logloss,color=color,alpha=.17,lw=.7)
   g=mm.groupby('fraction').logloss.mean();axes[0,col].plot(g.index*100,g.values,'o-',color=color,label='Non-EEG baseline' if model=='baseline' else NAMES[model])
   if model!='baseline':
    mm=inc[(inc.adjustment==0)&(inc.learner==learner)&(inc.model==model)]
    for rep,g in mm.groupby('repeat'):
     g=g.sort_values('fraction');axes[1,col].plot(g.fraction*100,g.delta_logloss,color=color,alpha=.17,lw=.7)
    g=mm.groupby('fraction').delta_logloss.mean();axes[1,col].plot(g.index*100,g.values,'o-',color=color)
  axes[0,col].set_title(LEARN[learner]);axes[1,col].axhline(0,color='.6',lw=.8);axes[1,col].set_xticks([50,75,100],['38','57','76–77']);axes[1,col].set_xlabel('Participants used for training')
 axes[0,0].set_ylabel('Held-out log loss');axes[1,0].set_ylabel('Log-loss increment');axes[0,0].legend(frameon=False,fontsize=9);fig.tight_layout();savefig(fig,'learning_curves')
 c.dump(OUT/'EMPIRICAL_VERIFICATION.json',{'jobs':len(jobs),'prediction_rows':len(preds),'paired_repeat_comparisons':len(inc),'fit_calls':sum(json.loads(f.read_text())['fit_calls'] for f in jobs),'independent_metric_check':True,'nested_subsamples_same_test_check':True,'original_logistic_reference_check':'passed within each full-size unadjusted job'})
def simulations():
 jobs=list((R/'results/simulation').glob('*/DONE.json'));assert len(jobs)==240
 met=pd.concat([pd.read_csv(f.parent/'METRICS.csv') for f in jobs],ignore_index=True);oracle=pd.concat([pd.read_csv(f.parent/'ORACLE.csv') for f in jobs],ignore_index=True);rows=[]
 for f in jobs:
  p=np.load(f.parent/'PREDICTIONS.npz');g=pd.read_csv(f.parent/'METRICS.csv')
  for _,row in g.iterrows():
   k=row.assessment+'__'+row.learner+'__'+row.model;m=independent_metrics(p['test_y'],p[k]);assert all(abs(m[v]-row[v])<1e-12 for v in m)
 keys=['scenario','replicate','train_n','assessment','learner']
 for key,g in met.groupby(keys):
  b=g[g.model=='baseline'].iloc[0];e=g[g.model=='eeg_augmented'].iloc[0]
  rows.append({**dict(zip(keys,key)),'baseline_logloss':b.logloss,'eeg_logloss':e.logloss,'delta_logloss':b.logloss-e.logloss,'delta_auroc':e.auroc-b.auroc,'delta_brier':b.brier-e.brier})
 inc=pd.DataFrame(rows);summary=[]
 for key,g in inc.groupby(['scenario','assessment','learner','train_n']):
  assert len(g)==20
  summary.append({**dict(zip(['scenario','assessment','learner','train_n'],key)),**{k:float(g[k].mean()) for k in ['baseline_logloss','eeg_logloss','delta_logloss','delta_auroc','delta_brier']},'mcse_delta_logloss':float(g.delta_logloss.std(ddof=1)/np.sqrt(20)),'positive_replicates':int(sum(g.delta_logloss>0))})
 summ=pd.DataFrame(summary)
 for name,data in [('SIMULATION_METRICS',met),('SIMULATION_INCREMENTS',inc),('SIMULATION_SUMMARY',summ),('SIMULATION_ORACLE',oracle)]:data.to_csv(OUT/(name+'.csv'),index=False,float_format='%.17g')
 panels=[('redundant','iid','Redundant EEG'),('complementary_linear','iid','Linear complementary EEG'),('complementary_nonlinear','iid','Nonlinear complementary EEG'),('acquisition_association','iid','Acquisition association: same distribution'),('acquisition_association','shifted','Acquisition association: shifted device')]
 fig,axes=plt.subplots(2,3,figsize=(9.0,6.4),sharex=True,sharey=True)
 for ax,(scenario,condition,title) in zip(axes.flat,panels):
  ax.axhline(0,color='.6',lw=.8)
  oo=oracle[(oracle.scenario==scenario)&(oracle.assessment==condition)].groupby('train_n').conditional_information_benchmark.mean();ax.plot(oo.index,oo.values,'--',color='#666666',label='Known-probability benchmark')
  for learner in ['logistic','rbf_svm']:
   g=summ[(summ.scenario==scenario)&(summ.assessment==condition)&(summ.learner==learner)].sort_values('train_n');ax.errorbar(g.train_n,g.delta_logloss,yerr=g.mcse_delta_logloss,color=COL[learner],marker='o',capsize=3,label=LEARN[learner]);raw=inc[(inc.scenario==scenario)&(inc.assessment==condition)&(inc.learner==learner)];ax.scatter(raw.train_n,raw.delta_logloss,s=9,color=COL[learner],alpha=.15)
  ax.set_title(title.replace('Acquisition association: ', 'Acquisition association:\n'),fontsize=10);ax.set_xticks([96,192,384]);ax.tick_params(axis='x',labelbottom=True);ax.set_xlabel('Synthetic training n')
 for ax in axes[:,0]:ax.set_ylabel('Log-loss increment')
 axes[1,2].axis('off');h,l=axes[0,0].get_legend_handles_labels();axes[1,2].legend(h,l,loc='center',frameon=False,fontsize=8);fig.tight_layout();savefig(fig,'controlled_simulations')
 c.dump(OUT/'SIMULATION_VERIFICATION.json',{'jobs':240,'independent_replicates_per_scenario':20,'fit_calls':sum(json.loads(f.read_text())['fit_calls'] for f in jobs),'saved_probability_metrics_independently_verified':True})
def external():
 p=R/'results/external';met=pd.read_csv(p/'METRICS.csv');pred=pd.read_csv(p/'PREDICTIONS.csv');inc=pd.read_csv(p/'INCREMENTS.csv')
 for key,g in pred.groupby(['baseline','learner','model']):
  row=met[(met.baseline==key[0])&(met.learner==key[1])&(met.model==key[2])].iloc[0];m=independent_metrics(g.y.to_numpy(int),g.p.to_numpy());assert all(abs(m[k]-row[k])<1e-12 for k in m)
 fig,ax=plt.subplots(figsize=(8.8,3.8));labels=[]
 for i,row in inc.iterrows():
  labels.append(('Task + eye' if row.baseline=='task_eye' else 'All non-EEG')+' / '+LEARN[row.learner]);v=row.delta_logloss;ax.errorbar(v,i,xerr=[[v-row.delta_logloss_conditional_low],[row.delta_logloss_conditional_high-v]],fmt='o',color=COL[row.learner],capsize=3)
 ax.axvline(0,color='.5',lw=.8);ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis();ax.set_xlabel('Test log-loss increment (baseline − EEG)');fig.tight_layout();savefig(fig,'independent_adult_increment')
 c.dump(OUT/'EXTERNAL_VERIFICATION.json',{'saved_probability_metrics_independently_verified':True,'test_participants':36,'paired_comparisons':6,'resampling_scope':'conditional on fitted models and observed test class counts'})
if __name__=='__main__':
 import sys
 for part in sys.argv[1:] or ['empirical','simulations','external']:globals()[part]()
