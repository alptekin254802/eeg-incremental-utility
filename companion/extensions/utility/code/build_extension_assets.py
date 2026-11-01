from pathlib import Path
import json,shutil
import pandas as pd
import run_extension as c
ROOT=c.ROOT
def build(root):
 out=Path(root);records=[]
 def n(value,signed=False):
  value=float(value);value=0 if abs(value)<.00005 else value
  z=f'{value:+.4f}' if signed and value!=0 else f'{value:.4f}'
  records.append({'value':float(value),'display':z});return '$'+z+'$'
 def write(name,s):
  start=len(records);(out/'tables'/name).write_text(s,encoding='utf-8')
 d=pd.read_csv(ROOT/'reports/EMPIRICAL_SUMMARY.csv');sim=pd.read_csv(ROOT/'reports/SIMULATION_SUMMARY.csv');adult=pd.read_csv(ROOT/'results/external/INCREMENTS.csv');am=pd.read_csv(ROOT/'results/external/METRICS.csv')
 learners={'logistic':'Logistic regression','rbf_svm':'RBF SVM','forest':'Random forest'};models={'regional_spectral':'Regional spectral','expanded_spectral':'Expanded spectral','covariance':'Spectral + covariance','entropy_hjorth':'Entropy + Hjorth','spectral_covariance_selector':'Spectral/covariance selector'}
 s=r'''\begin{table*}[t]
\centering
\caption{Matched learner sensitivity in the pediatric cohort. Each increment compares the indicated procedure with that learner's non-EEG baseline; values average ten dependent repeats. Positive differences favor EEG. These later comparisons retain the original outer partitions and are exploratory.}
\label{tab:learner}
\begin{tabular}{llrrrr}
\toprule
Learner & EEG procedure & Baseline LL & $\Delta$LL & $\Delta$AUROC & $\Delta$Brier\\
\midrule
'''
 for learner in learners:
  for model in ['spectral_covariance_selector','entropy_hjorth']:
   row=d[(d.fraction==1)&(d.adjustment==0)&(d.learner==learner)&(d.model==model)].iloc[0]
   s+=learners[learner]+' & '+('Library selector' if model.startswith('spectral') else models[model])+' & '+' & '.join([n(row.baseline_logloss),n(row.delta_logloss,True),n(row.delta_auroc,True),n(row.delta_brier,True)])+r'\\'+'\n'
 s+=r'\bottomrule\end{tabular}'+'\n'+r'\end{table*}'+'\n';write('main_learner.tex',s)
 s=r'''\begin{longtable}{llrrrrr}
\caption{Complete model-family comparisons. B and E denote baseline and EEG-augmented log loss. Positive increments favor EEG. U: unadjusted; DQ: device and EEG quality added to both models. All values average ten dependent repeats; supplementary machine-readable files retain their full distributions.}\label{tab:model-extension}\\
\toprule Learner/design & EEG procedure & B LL & E LL & $\Delta$LL & $\Delta$AUC & $\Delta$Brier\\\midrule\endfirsthead
\toprule Learner/design & EEG procedure & B LL & E LL & $\Delta$LL & $\Delta$AUC & $\Delta$Brier\\\midrule\endhead
\bottomrule\endfoot
'''
 short={'logistic':'Logistic','rbf_svm':'SVM','forest':'Forest'};mn={'regional_spectral':'Regional spectral','expanded_spectral':'Expanded spectral','covariance':'Covariance library','entropy_hjorth':'Entropy/Hjorth','spectral_covariance_selector':'Library selector'}
 for adj in [0,3]:
  for learner in learners:
   for model in models:
    g=d[(d.fraction==1)&(d.adjustment==adj)&(d.learner==learner)&(d.model==model)]
    if g.empty:continue
    row=g.iloc[0];s+=short[learner]+('/U' if adj==0 else '/DQ')+' & '+mn[model]+' & '+' & '.join([n(row.baseline_logloss),n(row.logloss),n(row.delta_logloss,True),n(row.delta_auroc,True),n(row.delta_brier,True)])+r'\\'+'\n'
 s+=r'\end{longtable}'+'\n';write('model_extension.tex',s)
 s=r'''\begin{table}[htbp]\centering
\caption{Training-size summaries for the two focal EEG procedures. The full-size endpoint reuses the unadjusted model-family analysis. B and E denote baseline and augmented log loss. Complete results for all five procedures appear in the accompanying CSVs.}\label{tab:learning-extension}
\begin{tabular}{llrrrrr}\toprule
Learner & Procedure & Training $n$ & B LL & E LL & $\Delta$LL & Positive/10\\\midrule
'''
 for learner in ['logistic','rbf_svm']:
  for model in ['spectral_covariance_selector','entropy_hjorth']:
   for frac in [.5,.75,1.]:
    row=d[(d.fraction==frac)&(d.adjustment==0)&(d.learner==learner)&(d.model==model)].iloc[0];nn=str(int(row.train_n_min)) if row.train_n_min==row.train_n_max else '76--77'
    s+=short[learner]+' & '+mn[model]+' & '+nn+' & '+' & '.join([n(row.baseline_logloss),n(row.logloss),n(row.delta_logloss,True),str(int(row.positive_repeats))])+r'\\'+'\n'
 s+=r'\bottomrule\end{tabular}\end{table}'+'\n';write('learning_extension.tex',s)
 s=r'''\begin{longtable}{llrrrr}
\caption{All controlled-simulation log-loss increments. MCSE is the Monte Carlo standard error across 20 independent replicates. Positive counts are descriptive and do not estimate test power. The independent test size is 4,096 for every replicate.}\label{tab:simulation-extension}\\
\toprule Scenario/assessment & Learner & Training $n$ & Mean $\Delta$LL & MCSE & Positive/20\\\midrule\endfirsthead
\toprule Scenario/assessment & Learner & Training $n$ & Mean $\Delta$LL & MCSE & Positive/20\\\midrule\endhead
\bottomrule\endfoot
'''
 scenario_names={'redundant':'Redundant','complementary_linear':'Linear complement','complementary_nonlinear':'Nonlinear complement','acquisition_association':'Acquisition'}
 for scenario,condition in [('redundant','iid'),('complementary_linear','iid'),('complementary_nonlinear','iid'),('acquisition_association','iid'),('acquisition_association','shifted')]:
  for learner in ['logistic','rbf_svm']:
   for size in [96,192,384]:
    row=sim[(sim.scenario==scenario)&(sim.assessment==condition)&(sim.learner==learner)&(sim.train_n==size)].iloc[0];label=scenario_names[scenario]+(' / shifted' if condition=='shifted' else (' / same' if scenario=='acquisition_association' else ''))
    s+=label+' & '+short[learner]+' & '+str(size)+' & '+n(row.delta_logloss,True)+' & '+n(row.mcse_delta_logloss)+' & '+str(int(row.positive_replicates))+r'\\'+'\n'
 s+=r'\end{longtable}'+'\n';write('simulation_extension.tex',s)
 s=r'''\begin{table}[htbp]\centering
\caption{Independent adult test results ($n=36$). B and E are baseline and EEG-augmented log loss; bracketed values are conditional 95\% test-resampling percentiles. Task/eye uses 49 baseline predictors; broader uses all 58 non-EEG predictors. E adds 18 spectral EEG predictors. All six pairs share the same test participants.}\label{tab:adult-extension}
\begin{tabular}{llrrrr}\toprule
Baseline & Learner & B LL & E LL & $\Delta$LL & Conditional range\\\midrule
'''
 for baseline in ['task_eye','all_non_eeg']:
  for learner in learners:
   row=adult[(adult.baseline==baseline)&(adult.learner==learner)].iloc[0];g=am[(am.baseline==baseline)&(am.learner==learner)];b=g[g.model=='baseline'].iloc[0];e=g[g.model=='eeg_augmented'].iloc[0]
   s+=('Task/eye' if baseline=='task_eye' else 'Broader')+' & '+short[learner]+' & '+' & '.join([n(b.logloss),n(e.logloss),n(row.delta_logloss,True),'['+n(row.delta_logloss_conditional_low,True)+', '+n(row.delta_logloss_conditional_high,True)+']'])+r'\\'+'\n'
 s+=r'\midrule Baseline & Learner & B AUROC & E AUROC & $\Delta$AUROC & $\Delta$Brier\\\midrule'+'\n'
 for baseline in ['task_eye','all_non_eeg']:
  for learner in learners:
   row=adult[(adult.baseline==baseline)&(adult.learner==learner)].iloc[0];g=am[(am.baseline==baseline)&(am.learner==learner)];b=g[g.model=='baseline'].iloc[0];e=g[g.model=='eeg_augmented'].iloc[0]
   s+=('Task/eye' if baseline=='task_eye' else 'Broader')+' & '+short[learner]+' & '+' & '.join([n(b.auroc),n(e.auroc),n(row.delta_auroc,True),n(row.delta_brier,True)])+r'\\'+'\n'
 s+=r'\bottomrule\end{tabular}\end{table}'+'\n';write('adult_extension.tex',s)
 for name in ['model_family_sensitivity','learning_curves','controlled_simulations','independent_adult_increment']:shutil.copy2(ROOT/'figures'/(name+'.pdf'),out/'figures'/(name+'.pdf'))
 c.dump(out/'audit/EXTENSION_ASSET_VALUES.json',{'source_sha256':{str(f.relative_to(ROOT)):c.sha(f) for f in [ROOT/'reports/EMPIRICAL_SUMMARY.csv',ROOT/'reports/SIMULATION_SUMMARY.csv',ROOT/'results/external/INCREMENTS.csv',ROOT/'results/external/METRICS.csv']},'numeric_values':records})
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--manuscript',required=True);build(p.parse_args().manuscript)
