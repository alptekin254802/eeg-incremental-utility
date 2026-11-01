"""Build the exploratory supplement assets from saved results only."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
parser=argparse.ArgumentParser()
parser.add_argument('--results',type=Path,default=Path(__file__).resolve().parents[1]/'results')
parser.add_argument('--manuscript',type=Path,required=True)
a=parser.parse_args();root=a.manuscript;data=a.results
for folder in ["tables","figures","audit"]:(root/folder).mkdir(parents=True,exist_ok=True)
df=pd.read_csv(data/'PRINCIPAL_CONTRASTS.csv',float_precision='round_trip')
groups=pd.read_csv(data/'GROUP_CONTRASTS.csv',float_precision='round_trip')
names={'spectral_covariance_selector':'Spectral/covariance selector','entropy_hjorth':'Entropy and Hjorth'}
directions={'X_to_Plus':r'X $\rightarrow$ Plus','Plus_to_X':r'Plus $\rightarrow$ X'}
ordered=[(d,m) for d in directions for m in names]
rows=[];cells=[]
def f(v):return f'{v:+.4f}'
for d,m in ordered:
 r=df[(df.direction==d)&(df.model==m)].iloc[0]
 values=[f(r.delta_ll),f'[{f(r.delta_ll_conditional_low)}, {f(r.delta_ll_conditional_high)}]',f(r.equal_class_delta_ll),f'[{f(r.equal_class_delta_ll_conditional_low)}, {f(r.equal_class_delta_ll_conditional_high)}]',f(r.delta_auroc),f(r.delta_brier)]
 cells+=values
 rows.append(names[m]+' & '+values[0]+' & '+values[1]+r' \\')
 rows.append(r'\quad Equal-class mean & '+values[2]+' & '+values[3]+r' \\')
text=r'''\begin{table}[htbp]
\centering
\caption{Exploratory device-separated transfer: log-loss changes and secondary metrics.}
\label{tab:transfer}
\begin{tabular}{lrr}
\toprule
Procedure & $\Delta$LL & Conditional 95\% range\\
\midrule
\multicolumn{3}{l}{Train EPOC X ($n=54$); assess EPOC+ ($n=42$)}\\
'''+ '\n'.join(rows[:4])+r'''
\midrule
\multicolumn{3}{l}{Train EPOC+ ($n=42$); assess EPOC X ($n=54$)}\\
'''+ '\n'.join(rows[4:])+r'''
\bottomrule
\end{tabular}

\vspace{6pt}
\begin{tabular}{llrr}
\toprule
Train $\rightarrow$ assess & Procedure & $\Delta$AUROC & $\Delta$Brier\\
\midrule
'''
for d,m in ordered:
 r=df[(df.direction==d)&(df.model==m)].iloc[0]
 text+=directions[d]+' & '+names[m]+' & '+f(r.delta_auroc)+' & '+f(r.delta_brier)+r' \\'+'\n'
text+=r'''\bottomrule
\end{tabular}
\par\smallskip
\begin{minipage}{\textwidth}\small
Positive differences favor EEG. Overall log-loss means use the observed target composition; equal-class means weight ADHD and controls equally. Ranges resample target participants within groups with fixed source fits and class counts (5,000 draws); they omit source-training and selection uncertainty. The selector chose regional spectral features in both directions. The addendum retains absolute metrics, all fixed candidates, and conditional draws for all four principal comparisons.
\end{minipage}
\end{table}
'''
(root/'tables/supp_transfer.tex').write_text(text,encoding='utf-8')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
fig,axs=plt.subplots(2,1,figsize=(7.0,4.7),sharex=True,layout='constrained')
for ax,(d,title) in zip(axs,[('X_to_Plus','Train EPOC X; assess EPOC+ (25 ADHD, 17 control)'),('Plus_to_X','Train EPOC+; assess EPOC X (16 ADHD, 38 control)')]):
 ax.axvline(0,color='#737373',lw=.8,zorder=0)
 for y,m in enumerate(names):
  r=df[(df.direction==d)&(df.model==m)].iloc[0]
  g=groups[(groups.direction==d)&(groups.model==m)].set_index('group')
  points=[(g.loc[0,'delta_ll'],'o','#24618c','Control mean'),(g.loc[1,'delta_ll'],'s','#b85136','ADHD mean'),(r.delta_ll,'D','#222222','Observed-composition mean'),(r.equal_class_delta_ll,'X','#168577','Equal-class mean')]
  for val,marker,color,label in points:ax.scatter(val,y,marker=marker,s=48,color=color,label=label if y==0 else None,zorder=3)
 ax.set_yticks([0,1],list(names.values()));ax.set_ylim(1.6,-.6)
 ax.set_title(title,fontsize=10,loc='left',pad=10)
 ax.grid(axis='x',color='#eeeeee');ax.spines[['top','right','left']].set_visible(False)
 ax.tick_params(axis='y',length=0)
axs[-1].set_xlabel('Log-loss change (baseline minus EEG-augmented model)')
axs[-1].set_xlim(-.10,.16)
handles,labels=axs[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False,fontsize=9)
fig.savefig(root/'figures/device_transfer.pdf',metadata={'CreationDate':None,'ModDate':None})
fig.savefig(root/'figures/device_transfer.png',dpi=200)
(root/'audit/TRANSFER_ASSET_VALUES.json').write_text(json.dumps({'results_sha256':hashlib.sha256((data/'PRINCIPAL_CONTRASTS.csv').read_bytes()).hexdigest(),'displayed_values':cells,'row_order':ordered,'figure_group_points':groups[groups.model.isin(names)].to_dict('records')},indent=2),encoding='utf-8')
print('Created table S9 and figure S4 from saved results.')
