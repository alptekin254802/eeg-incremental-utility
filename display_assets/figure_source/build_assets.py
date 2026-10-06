"""Render article tables and figures from the accompanying saved results.

This command does not train models or generate signal controls.
Run from any directory with Python, pandas, NumPy, and Matplotlib installed.
"""
from pathlib import Path
import hashlib
import json
import argparse
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SOURCE_ASSETS = Path(__file__).resolve().parents[1]
SOURCES = {}
CELLS = []
PLOTS = {}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--release-root', type=Path, default=SOURCE_ASSETS.parent, help='Repository containing the saved result files.')
parser.add_argument('--output', type=Path, required=True, help='Display output directory outside the repository.')
parser.add_argument('--tables-only', action='store_true', help='Regenerate tables and value records while preserving existing PDF/SVG figure assets.')
parser.add_argument('--figure', choices=['repeat_increments','acquisition_phase_controls','component_increments','calibration','participant_changes','coefficients'], help='Regenerate only the named figure, preserving the other figure files.')
ARGS = parser.parse_args()
if ARGS.tables_only and ARGS.figure:
    parser.error('--tables-only and --figure cannot be combined')
RELEASE = ARGS.release_root.resolve()
ROOT = ARGS.output.resolve()
if ROOT.is_relative_to(RELEASE):
    parser.error('--output must be outside the repository')
TABLES = ROOT / 'tables'
FIGURES = ROOT / 'figures'
for directory in [TABLES, FIGURES, ROOT / 'figure_source']:
    directory.mkdir(parents=True, exist_ok=True)
for name in ['main_cohort.tex', 'cohort.tex', 'identifier_crosswalk.tex']:
    shutil.copy2(SOURCE_ASSETS / 'tables' / name, TABLES / name)
if ARGS.tables_only or ARGS.figure:
    for source in (SOURCE_ASSETS / 'figures').iterdir():
        if source.suffix in {'.pdf', '.svg'} and not (FIGURES / source.name).exists():
            shutil.copy2(source, FIGURES / source.name)


def read(name):
    p = RELEASE/name
    SOURCES[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return pd.read_csv(p)

def value(x, source, key, signed=False):
    places = 5 if source == 'phase' else 4
    s = format(float(x), ('+' if signed else '') + f'.{places}f')
    CELLS.append(dict(source=source, key=key, value=float(x), display=s))
    return '$'+s+'$' if signed else s

def tex(name, text):
    (TABLES/name).write_text(text+'\n', encoding='utf-8')

def start_table(caption, label, cols, head, wide=False):
    env = 'table*' if wide else 'table'
    return '\\begin{'+env+'}[!htbp]\n\\centering\\small\n\\caption{'+caption+'}\n\\label{'+label+'}\n'+r'\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.14}'+'\n'+r'\begin{tabularx}{\textwidth}{@{}>'+r'{\raggedright\arraybackslash}X'+cols+'@{}}\n'+r'\toprule'+'\n'+head+r'\\\midrule'+'\n'

def end_table(note='', wide=False):
    return r'\bottomrule\end{tabularx}'+'\n'+(r'\par\smallskip\begin{minipage}{\textwidth}\footnotesize '+note+r'\end{minipage}'+'\n' if note else '')+'\\end{'+('table*' if wide else 'table')+'}\n'

primary = read('audit/preregistered_analysis/PRIMARY_REPEAT_SUMMARY.csv')
master = read('audit/preregistered_analysis/PREREGISTERED_RESULTS_MASTER_TABLE.csv')
explore = pd.concat([read('audit/exploratory_eeg_battery/REPEAT_LEVEL_RESULTS.csv'),
                     read('results/e5/REPEAT_RESULTS.csv')], ignore_index=True)
post = read('expected/posthoc/results/comparison_summary.csv').set_index('comparison')
post_repeat = read('expected/posthoc/results/repeat_results.csv')
controls = read('expected/posthoc/signal_controls/comparison_summary.csv').set_index('comparison')
ladder = read('audit/preregistered_analysis/MODALITY_LADDER_CONSOLIDATION.csv')
diagnosis = read('audit/preregistered_analysis/DIAGNOSED_SENSITIVITY.csv')
cal = read('audit/preregistered_analysis/CALIBRATION_RESULTS.csv')
cal_numerical = read('results/numerical_followups/CALIBRATION_NUMERICAL.csv')
fixed = {f: read(name) for f, name in [('E0','audit/preregistered_analysis/E0_CORE_SENSITIVITY.csv'),('E1','audit/preregistered_analysis/FIXED_E1_SECONDARY.csv'),('E2','audit/preregistered_analysis/FIXED_E2_SECONDARY.csv')]}
ladder_repeat = read('audit/preregistered_analysis/EXPLORATORY_MODALITY_LADDER.csv')
# Use saved repeat precision rather than the shortened consolidation values.
for idx, row in ladder.iterrows():
    nxt = ladder_repeat[ladder_repeat.model == row.to_model].sort_values('repeat')
    if row.row_type == 'absolute':
        for metric in ['logloss','auroc','brier']: ladder.loc[idx,metric] = nxt[metric].mean()
    else:
        prev = ladder_repeat[ladder_repeat.model == row.from_model].sort_values('repeat')
        for metric in ['logloss','auroc','brier']:
            sign = 1 if metric == 'auroc' else -1
            ladder.loc[idx,'delta_'+metric+'_favoring_next'] = sign*(nxt[metric].to_numpy()-prev[metric].to_numpy()).mean()

names = {
    'adaptive':'Primary EEG-augmented procedure', 'E0':'Regional spectral features',
    'E1':'Expanded spectral features', 'E2':'Expanded spectral and covariance features',
    'E3':'Phase-lag connectivity', 'E4':'Permutation entropy and Hjorth features',
    'E5':'Spectral-state features', 'M0':'Demographics',
    'M1':'Demographics and task behavior', 'M2':'Non-EEG baseline',
}
baseline = [primary['m2_'+x].mean() for x in ['logloss','auroc','brier']]
models = {}
models['adaptive'] = [primary['m3_'+x].mean() for x in ['logloss','auroc','brier']]
for family in ['E0','E1','E2']:
    models[family] = [fixed[family]['representation_'+m].mean() for m in ['logloss','auroc','brier']]
for family in ['E3','E4','E5']:
    d=explore[explore.family==family]
    models[family]=[d['eeg_'+x].mean() for x in ['logloss','auroc','brier']]

def delta(family):
    if family in ['adaptive','E0','E1','E2']:
        key='M2_vs_original_'+family
        r=post.loc[key]
        return [r.delta_ll,r.delta_auroc,r.delta_brier]
    d=explore[explore.family==family]
    return [d[x].mean() for x in ['delta_ll','delta_auroc','delta_brier']]

t=start_table('Participant-level prediction with and without EEG.', 'tab:performance', 'rrrr', r'Model & Log loss & $\Delta_{\mathrm{LL}}$ & AUROC & Brier',True)
t+=r'\multicolumn{5}{l}{\textit{Descriptive non-EEG models}}\\'+'\n'
for f in ['M0','M1','M2']:
    r=ladder[(ladder.row_type=='absolute')&(ladder.to_model==f)].iloc[0]
    t+=names[f]+' & '+' & '.join([value(r.logloss,'ladder',f+' LL'),'---',value(r.auroc,'ladder',f+' AUROC'),value(r.brier,'ladder',f+' Brier')])+r'\\'+'\n'
for heading, families in [('Preregistered primary comparison',['adaptive']),('Prespecified sensitivity and secondary representations',['E0','E1','E2']),('Post-hoc representations',['E3','E4','E5'])]:
    t+=r'\midrule\multicolumn{5}{l}{\textit{'+heading+r'}}\\'+'\n'
    for f in families:
        v=models[f]; dd=delta(f)
        t+=names[f]+' & '+' & '.join([value(v[0],f,'LL'),value(dd[0],f,'delta_ll',True),value(v[1],f,'AUROC'),value(v[2],f,'Brier')])+r'\\'+'\n'
t+=end_table(r'The non-EEG models successively add task behavior and object gaze to demographics; no differences are displayed for these descriptive rows. All EEG models retain the full non-EEG baseline predictors, including object gaze. Values average ten dependent repeats in the same 96 participants. Lower log loss and Brier score and higher AUROC indicate better performance. $\Delta_{\mathrm{LL}}$ is full non-EEG baseline minus EEG-augmented loss; positive values favor EEG. Differences are calculated before display rounding. The primary procedure selects within the initial library; regional spectral features provide its required fixed-representation sensitivity, and later families are evaluated separately.',True)
tex('main_performance.tex',t)

t=start_table('EEG increments under matched device and signal-quality adjustment.', 'tab:adjustment', 'rrr', r'EEG procedure or representation & $\Delta_{\mathrm{LL}}$ & $\Delta_{\mathrm{AUROC}}$ & $\Delta_{\mathrm{Brier}}$',True)
for heading, keys in [
    ('No adjustment', ['M2_vs_original_adaptive','M2_vs_E4']),
    ('Device adjustment', ['M2_D_vs_adaptive_D','M2_D_vs_E4_D']),
    ('Device and EEG-quality adjustment', ['M2_DQ_vs_adaptive_DQ','M2_DQ_vs_E4_DQ'])
]:
    if heading != 'No adjustment': t+=r'\midrule'+'\n'
    t+=r'\multicolumn{4}{l}{\textit{'+heading+r'}}\\'+'\n'
    for key, label in zip(keys, ['Primary EEG procedure','Full entropy and Hjorth model']):
        r=post.loc[key]
        t+=label+' & '+' & '.join(value(r[m],'posthoc',key+'/'+m,True) for m in ['delta_ll','delta_auroc','delta_brier'])+r'\\'+'\n'
t+=end_table(r'Positive differences favor EEG augmentation: baseline minus augmented log loss or Brier score, and augmented minus baseline AUROC. Each comparison adds the named covariates to both the baseline and EEG-augmented model. EEG-quality covariates are clean-window fraction and interpolated-channel count; this adjusted baseline requires EEG acquisition. The unadjusted primary comparison is preregistered; the entropy and Hjorth and all adjusted comparisons are post hoc. Values average the same ten dependent repeats.',True)
tex('main_adjustment.tex',t)

t=start_table('Complete mean performance and supporting non-EEG comparisons.','tab:full-performance','rrrrrr',r'Model & LL & $\Delta_{\mathrm{LL}}$ & AUROC & $\Delta_{\mathrm{AUROC}}$ & Brier & $\Delta_{\mathrm{Brier}}$')
for f in ['M0','M1','M2']:
    r=ladder[(ladder.row_type=='absolute')&(ladder.to_model==f)].iloc[0]
    t+=names[f]+' & '+' & '.join([value(r.logloss,'ladder',f+' LL'),'---',value(r.auroc,'ladder',f+' AUROC'),'---',value(r.brier,'ladder',f+' Brier'),'---'])+r'\\'+'\n'
for heading,fs in [('Preregistered primary',['adaptive']),('Prespecified sensitivity and secondary',['E0','E1','E2']),('Post hoc',['E3','E4','E5'])]:
    t+=r'\midrule\multicolumn{7}{l}{\textit{'+heading+r'}}\\'+'\n'
    for f in fs:
        v=models[f];dd=delta(f);cells=[]
        for i,m in enumerate(['LL','AUROC','Brier']):cells.extend([value(v[i],f,m),value(dd[i],f,'delta '+m,True)])
        t+=names[f]+' & '+' & '.join(cells)+r'\\'+'\n'
t+=r'\midrule\multicolumn{7}{l}{\textit{Descriptive non-EEG additions; differences favor the later model}}\\'+'\n'
trans=ladder[ladder.row_type!='absolute']
for _,r in trans.iterrows():
    title='Add task behavior' if r.from_model=='M0' else 'Add object gaze'
    t+=title+' & --- & '+value(r.delta_logloss_favoring_next,'ladder',title+' deltaLL',True)+' & --- & '+value(r.delta_auroc_favoring_next,'ladder',title+' deltaAUROC',True)+' & --- & '+value(r.delta_brier_favoring_next,'ladder',title+' deltaBrier',True)+r'\\'+'\n'
t+=end_table(r'LL: log loss. EEG differences use the non-EEG baseline. Positive differences favor the evaluated addition; repeat estimates are descriptive. Expanded spectral models include the preceding EEG representation. Full precision and repeat-specific estimates are supplied as CSV files.')
tex('full_performance.tex',t)

t=start_table('Diagnosis-status sensitivity: mean results in the 84-participant subset.','tab:diagnosis','rrr',r'Model or comparison & Log loss & AUROC & Brier')
for code,name in [('m2','Non-EEG baseline'),('m3','Primary EEG procedure')]:
    t+=name+' & '+' & '.join(value(diagnosis[code+'_'+m].mean(),'diagnosis',code+' '+m) for m in ['logloss','auroc','brier'])+r'\\'+'\n'
t+=r'\midrule EEG increment'+' & '+' & '.join(value(diagnosis[m].mean(),'diagnosis',m,True) for m in ['delta_ll','delta_auroc','delta_brier'])+r'\\'+'\n'
t+=end_table('The last row gives oriented differences, not absolute performance. The target is exact diagnosed yes/no, with new stratified folds; these results are not an independent replication.')
tex('diagnosis.tex',t)

t=r'''\begin{table}[!htbp]
\centering\small
\caption{Scalar calibration estimates from the numerical follow-up.}\label{tab:calibration}
\begin{tabular}{r rrr rrr}
\toprule
& \multicolumn{3}{c}{Non-EEG baseline} & \multicolumn{3}{c}{Primary EEG procedure}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Repeat & Intercept & Slope & CITL & Intercept & Slope & CITL\\\midrule
'''
for rep in range(10):
    cells=[]
    for model in ['M2','adaptive_M3']:
        r=cal_numerical[(cal_numerical.model==model)&(cal_numerical.repeat==rep)].iloc[0]
        marker=r'$^{\dagger}$' if r.original_failure else ''
        cells.extend(value(r[m],'calibration_numerical',f'{model}/{rep}/{m}',m!='slope')+(marker if m=='slope' else '') for m in ['intercept','slope','citl'])
    t+=str(rep+1)+' & '+' & '.join(cells)+r'\\'+'\n'
t+=r'''\bottomrule\end{tabular}
\par\smallskip\begin{minipage}{\textwidth}\footnotesize CITL: calibration-in-the-large. Repeat numbers start at one here and at zero in the data. $\dagger$ marks an original BFGS precision-loss failure. Values are the post-hoc Newton solutions of the same unpenalized calibration model; the original estimates and failure records are retained in the data package. The predictive probabilities are unchanged.\end{minipage}
\end{table}'''
tex('calibration.tex',t)

component_names={'E4_only_perm_entropy':'Permutation entropy only','E4_only_hjorth_mobility':'Hjorth mobility only','E4_only_hjorth_complexity':'Hjorth complexity only','E4_without_perm_entropy':'Without permutation entropy','E4_without_hjorth_mobility':'Without Hjorth mobility','E4_without_hjorth_complexity':'Without Hjorth complexity','E0_relative9':'Regional relative power (nine features)','E0_PCA9':'Regional spectral PCA (nine components)'}
groups=[('Reference comparisons: non-EEG baseline', ['M2_vs_original_'+x for x in ['adaptive','E0','E1','E2']]+['M2_vs_E4']),
        ('Adding acquisition covariates to the non-EEG baseline',['M2_vs_M2_D','M2_vs_M2_DQ']),
        ('Device-adjusted baseline and EEG model',['M2_D_vs_'+x+'_D' for x in ['E0','E1','E2','E4','adaptive']]),
        ('Device-plus-quality-adjusted baseline and EEG model',['M2_DQ_vs_'+x+'_DQ' for x in ['E0','E1','E2','E4','adaptive']]),
        ('Components and spectral controls: non-EEG baseline',['M2_vs_'+x for x in component_names]),
        ('Direct contrasts: full entropy and Hjorth model as baseline',['E4_vs_'+x for x in component_names]),
        ('Selection including the option to omit EEG',['M2_vs_optout','original_adaptive_vs_optout'])]
assert set(k for _,ks in groups for k in ks)==set(post.index)
t=r'''\begingroup\small
\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.06}
\begin{longtable}{@{}p{0.47\textwidth}rrrr@{}}
\caption{All post-hoc explanatory comparisons and reference results.}\label{tab:comparisons}\\
\toprule Evaluated model or addition & $\Delta_{\mathrm{LL}}$ & $\Delta_{\mathrm{AUROC}}$ & $\Delta_{\mathrm{Brier}}$ & $R_+$\\\midrule
\endfirsthead
\multicolumn{5}{l}{Table \thetable\ (continued)}\\
\toprule Evaluated model or addition & $\Delta_{\mathrm{LL}}$ & $\Delta_{\mathrm{AUROC}}$ & $\Delta_{\mathrm{Brier}}$ & $R_+$\\\midrule
\endhead
\midrule\multicolumn{5}{r}{Continued on next page}\\\endfoot
\bottomrule\endlastfoot
'''
for heading,keys in groups:
    t+=r'\multicolumn{5}{@{}l}{\textit{'+heading+r'}}\\*'+'\n'
    for key in keys:
        r=post.loc[key];code=r.augmented
        label=component_names.get(code)
        if label is None:
            basecode=code.replace('original_','').replace('_DQ','').replace('_D','')
            label=names.get(basecode,code)
        if key=='M2_vs_M2_D':label='Add device indicator'
        if key=='M2_vs_M2_DQ':label='Add device and EEG-quality measures'
        if key=='M2_vs_optout':label='EEG-optional selector vs non-EEG baseline'
        if key=='original_adaptive_vs_optout':label='EEG-optional selector vs primary procedure'
        t+=label+' & '+' & '.join(value(r[m],'posthoc',key+'/'+m,True) for m in ['delta_ll','delta_auroc','delta_brier'])+' & '+str(int(r.positive_repeats))+(r'\\*' if key != keys[-1] else r'\\')+'\n'
    t+=r'\addlinespace'+'\n'
t+=r'''\end{longtable}
\begin{minipage}{\textwidth}\footnotesize Positive differences favor the evaluated model relative to the baseline named in each block. $R_+$ counts favorable log-loss differences among ten dependent repeats. The first block repeats existing results as references for the later comparisons. Device/quality covariates enter both paired models. Direct contrasts against the full entropy and Hjorth model compare separately refitted models. Absolute losses and repeat-level estimates are supplied in the accompanying data.\end{minipage}
\endgroup'''
tex('all_comparisons.tex',t)

t=r'''\begin{table}[!htbp]
\centering\small
\caption{All Fourier-phase control comparisons.}\label{tab:phase}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{r rrr rrr}
\toprule
& \multicolumn{3}{c}{Control versus non-EEG baseline} & \multicolumn{3}{c}{Observed model versus control}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Control & $\Delta_{\mathrm{LL}}$ & $\Delta_{\mathrm{AUROC}}$ & $\Delta_{\mathrm{Brier}}$ & $\Delta_{\mathrm{LL}}$ & $\Delta_{\mathrm{AUROC}}$ & $\Delta_{\mathrm{Brier}}$\\\midrule
'''
for i in range(20):
    vals=[]
    for key in [f'M2_vs_surrogate_{i:02d}',f'surrogate_{i:02d}_vs_observed_E4']:
        r=controls.loc[key];vals.extend(value(r[m],'phase',key+'/'+m,True) for m in ['delta_ll','delta_auroc','delta_brier'])
    t+=str(i+1)+' & '+' & '.join(vals)+r'\\'+'\n'
t+=r'''\bottomrule\end{tabular}
\par\smallskip\begin{minipage}{\textwidth}\footnotesize Positive values favor the control in the left block and the observed entropy and Hjorth model in the right block. Every control uses the same participants and partitions. Indices 1--20 correspond to 0--19 in the data. Mean differences are descriptive, not independent replications or significance tests.\end{minipage}
\end{table}'''
tex('phase_comparisons.tex',t)

ranges = read('results/numerical_followups/FIXED_OOF_RANGES.csv')
t=start_table('Conditional descriptive fixed-OOF resampling ranges.','tab:fixed-oof','rrr',r'Model & Mean $\Delta_{\mathrm{LL}}$ & Lower & Upper')
for _,r in ranges.iterrows():
    label='Primary EEG procedure' if r.model=='original_adaptive' else 'Entropy and Hjorth'
    t+=label+' & '+' & '.join(value(r[m],'fixed_oof',r.model+'/'+m,True) for m in ['mean_delta_ll','lower','upper'])+r'\\'+'\n'
t+=end_table(r'Post-hoc descriptive resampling of 96 paired participant mean losses, using 5,000 draws and the 2.5th and 97.5th percentiles. Fits, partitions, selections, and predictions remain fixed. These are not confidence intervals for the learning procedure; no $p$ values are calculated.')
tex('fixed_oof_ranges.tex',t)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'pdf.fonttype':42,'svg.fonttype':'none','savefig.dpi':180})
BLUE='#355c83';TEAL='#16786a';GREY='#657786';GOLD='#a96a24'
def save(fig,name,records):
    if not ARGS.tables_only and (ARGS.figure is None or ARGS.figure == name):
        fig.savefig(FIGURES/(name+'.pdf'),bbox_inches='tight')
        fig.savefig(FIGURES/(name+'.svg'),bbox_inches='tight')
    else:
        for suffix in ['.pdf','.svg']:
            if not (FIGURES/(name+suffix)).is_file():
                raise FileNotFoundError(FIGURES/(name+suffix))
    plt.close(fig)
    PLOTS[name]=records
def clean(ax):
    ax.axhline(0,color='#8c989f',lw=.7,zorder=0)
    ax.grid(axis='y',color='#e5e9ec',lw=.55,zorder=0)
    ax.tick_params(labelsize=8,length=3)

fig,axes=plt.subplots(2,2,figsize=(7.05,4.6),sharex=True,sharey=True,layout='constrained')
plotrows=[]
repeat_panel_titles=['Primary EEG procedure','Phase-lag connectivity','Entropy and Hjorth','Spectral-state features']
for ax,f,title,color in zip(axes.flat,['adaptive','E3','E4','E5'],repeat_panel_titles,[BLUE,GREY,TEAL,GOLD]):
    vals=primary.delta_ll.to_numpy() if f=='adaptive' else explore[explore.family==f].sort_values('repeat').delta_ll.to_numpy()
    ax.scatter(range(1,11),vals,s=23,color=color,zorder=3)
    ax.scatter([11.3],[vals.mean()],marker='D',s=37,color=color,zorder=4)
    ax.set_title(title,fontsize=9.5,loc='left');clean(ax)
    ax.set_xlim(.4,12);ax.set_ylim(-.14,.09);ax.set_xticks([1,5,10,11.3],['1','5','10','Mean'])
    plotrows.append(dict(model=f,repeat_values=vals.tolist(),mean=float(vals.mean())))
fig.supylabel('Log-loss increment (positive favors EEG)',fontsize=10)
fig.supxlabel('Cross-validation repeat',fontsize=9)
save(fig,'repeat_increments',plotrows)

fig,axes=plt.subplots(1,2,figsize=(7.05,3.15),layout='constrained',gridspec_kw={'width_ratios':[1.08,1]})
records=[]
for f,label,color,marker in [('original_adaptive','Primary EEG procedure',BLUE,'s'),('E4','Entropy and Hjorth',TEAL,'o')]:
    keys=['M2_vs_'+f]+[f'M2_{d}_vs_{"adaptive" if f=="original_adaptive" else f}_{d}' for d in ['D','DQ']]
    vals=[float(post.loc[k,'delta_ll']) for k in keys]
    axes[0].plot(range(3),vals,color=color,marker=marker,lw=1.1,label=label,markersize=4.5);records.append(dict(keys=keys,values=vals))
axes[0].set_xticks(range(3),['No\nadjustment','Device','Device +\nEEG quality']);axes[0].set_ylim(-.045,.04)
axes[0].legend(frameon=False,fontsize=7.8,loc='lower left');axes[0].set_title('a  Matched acquisition adjustment',loc='left',fontsize=9.5);clean(axes[0])
vals=np.array([controls.loc[f'M2_vs_surrogate_{i:02d}','delta_ll'] for i in range(20)])
axes[1].scatter(range(1,21),vals,s=21,color=GREY)
observed=float(post.loc['M2_vs_E4','delta_ll']);axes[1].axhline(observed,color=TEAL,lw=1.15,label='Observed entropy and Hjorth')
axes[1].legend(frameon=False,fontsize=7.4,loc='lower center')
axes[1].set_xticks([1,5,10,15,20]);axes[1].set_xlabel('Phase-control dataset');axes[1].set_ylim(.0205,.032)
axes[1].set_title('b  Window-wise phase controls',loc='left',fontsize=9.5);axes[1].grid(axis='y',color='#e5e9ec',lw=.55)
for ax in axes:ax.set_ylabel('Mean log-loss increment',fontsize=8.5);ax.tick_params(labelsize=8)
save(fig,'acquisition_phase_controls',dict(adjustments=records,phase_values=vals.tolist(),observed=observed))

fig,axes=plt.subplots(2,5,figsize=(7,4),sharex=True,sharey=True)
for rep,ax in enumerate(axes.flat):
    ax.plot([0,1],[0,1],ls='--',lw=.65,color=GREY)
    for model,label,color,marker in [('M2','Non-EEG baseline',BLUE,'o'),('adaptive_M3','Primary EEG procedure',GOLD,'s')]:
        v=cal[(cal.record_type=='reliability_bin')&(cal.model==model)&(cal.repeat==rep)].sort_values('bin')
        ax.plot(v.mean_probability,v.observed_rate,color=color,marker=marker,markersize=3,lw=.9,label=label)
    ax.set_title('Repeat '+str(rep+1),fontsize=9);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_xticks([0,.5,1]);ax.set_yticks([0,.5,1]);ax.grid(color='#e5e9ec',lw=.5);ax.tick_params(labelsize=7)
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False,fontsize=9)
fig.supxlabel('Mean predicted probability',fontsize=9);fig.supylabel('Observed study-group frequency',fontsize=9)
fig.subplots_adjust(top=.83,bottom=.14,left=.09,right=.99,wspace=.22,hspace=.4)
save(fig,'calibration',cal[cal.record_type=='reliability_bin'].fillna('').to_dict('records'))

order=['E4']+list(component_names)
fig,ax=plt.subplots(figsize=(7,4.4),layout='constrained')
records=[]
for i,f in enumerate(order):
    vals=post_repeat[post_repeat.comparison=='M2_vs_'+f].sort_values('repeat').delta_ll.to_numpy()
    offsets=np.linspace(-.18,.18,len(vals))
    ax.scatter(vals,i+offsets,s=14,color=TEAL if f=='E4' else GREY,alpha=.75)
    ax.scatter([vals.mean()],[i],marker='D',s=31,color=TEAL if f=='E4' else BLUE,zorder=4)
    records.append(dict(model=f,repeat_values=vals.tolist()))
ax.set_yticks(range(len(order)),['Full entropy and Hjorth']+[component_names[f].replace(' (nine features)',' (9 features)').replace(' (nine components)',' (9 components)') for f in component_names],fontsize=8.5)
ax.invert_yaxis();ax.axvline(0,color=GREY,lw=.7);ax.grid(axis='x',color='#e5e9ec',lw=.6);ax.set_xlabel('Log-loss increment relative to non-EEG baseline')
save(fig,'component_increments',records)

participant_losses = read('expected/posthoc/results/participant_loss_diagnostics.csv')
participant_rows = []
for model in ['original_adaptive', 'E4']:
    ordered = participant_losses[participant_losses.model == model].sort_values('mean_delta_ll').copy()
    ordered['independent_rank'] = np.arange(1, len(ordered) + 1)
    participant_rows.append(ordered[['model', 'independent_rank', 'mean_delta_ll']])
participant = pd.concat(participant_rows, ignore_index=True)
fig,axes=plt.subplots(2,1,figsize=(7,4.25),sharex=True,sharey=True,layout='constrained')
for ax,f,label in zip(axes,['original_adaptive','E4'],['Primary EEG procedure','Permutation entropy and Hjorth']):
    v=participant[participant.model==f].sort_values('independent_rank')
    ax.scatter(v.independent_rank,v.mean_delta_ll,s=10,c=np.where(v.mean_delta_ll>0,TEAL,GOLD))
    ax.set_title(label,loc='left',fontsize=9.5);clean(ax);ax.set_ylabel('Mean loss difference')
axes[1].set_xlabel('Participant rank within each model (independent sorting)')
save(fig,'participant_changes',participant.to_dict('records'))

coef=read('expected/posthoc/results/coefficient_summary.csv');coef=coef[(coef.model=='E4') & (~coef.feature.isin(['age','gender','work_speed','omission_errors','commission_errors','gaze_entropy','gaze_dispersion']))]
features=[reg+'_'+m for reg in ['frontal','temporal','posterior'] for m in ['perm_entropy','hjorth_mobility','hjorth_complexity']]
coef=coef.set_index('feature').loc[features]
fig,ax=plt.subplots(figsize=(7,4.05),layout='constrained')
for i,(f,v) in enumerate(coef.iterrows()):
    ax.plot([v['min'],v['max']],[i,i],color=GREY,lw=1.5);ax.scatter(v['median'],i,s=25,color=TEAL,zorder=3)
labels=[f.replace('_perm_entropy',': permutation entropy').replace('_hjorth_mobility',': mobility').replace('_hjorth_complexity',': complexity').capitalize() for f in features]
ax.set_yticks(range(9),labels,fontsize=9);ax.invert_yaxis();ax.axvline(0,color=GREY,lw=.7);ax.grid(axis='x',color='#e5e9ec',lw=.6);ax.set_xlabel('Standardized coefficient: median and full range')
save(fig,'coefficients',coef.reset_index().to_dict('records'))

locations = {
    'tables/main_cohort.tex':'Main Table 1', 'tables/main_performance.tex':'Main Table 2',
    'tables/main_adjustment.tex':'Main Table 3', 'tables/cohort.tex':'Supplementary Table S1',
    'tables/identifier_crosswalk.tex':'Supplementary Table S2 (identifier crosswalk)',
    'tables/full_performance.tex':'Supplementary Table S3', 'tables/diagnosis.tex':'Supplementary Table S4',
    'tables/calibration.tex':'Supplementary Table S5', 'tables/all_comparisons.tex':'Supplementary Table S6',
    'tables/phase_comparisons.tex':'Supplementary Table S7',
    'tables/fixed_oof_ranges.tex':'Supplementary Table S8',
    'repeat_increments':'Main Figure 1', 'acquisition_phase_controls':'Main Figure 2',
    'component_increments':'Main Figure 3', 'calibration':'Supplementary Figure S1',
    'participant_changes':'Supplementary Figure S2', 'coefficients':'Supplementary Figure S3'
}
static_hashes={name:hashlib.sha256((TABLES/name).read_bytes()).hexdigest() for name in ['main_cohort.tex','cohort.tex']}
presentation={'repeat_increments':{'panel_titles':repeat_panel_titles,'status_explanation':'Main Figure 1 caption'}}
(ROOT/'figure_source/ASSET_VALUES.json').write_text(json.dumps(dict(source_sha256=SOURCES,table_cells=CELLS,plots=PLOTS,display_locations=locations,static_table_sha256=static_hashes,presentation=presentation),indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({'generated_tables':8,'figure_assets_preserved':ARGS.tables_only,'selected_figure':ARGS.figure,'figures':6,'source_files':len(SOURCES),'numeric_table_cells':len(CELLS)}))
