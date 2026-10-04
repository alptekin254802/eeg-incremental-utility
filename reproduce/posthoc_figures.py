"""Standalone scientific figures. Points/ranges describe dependent CV fits, not CIs."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
HERE=Path(os.environ["EEG_RUN_ROOT"])/"posthoc"
RESULT=HERE/"results";FIG=HERE/"figures";FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False,
                    "pdf.fonttype":42,"ps.fonttype":42,"savefig.dpi":220})
BLUE="#126782";ORANGE="#C76B35";GREY="#A8B3BA";GREEN="#34745C"
rr=pd.read_csv(RESULT/"repeat_results.csv");ss=pd.read_csv(RESULT/"comparison_summary.csv").set_index("comparison")

def save(fig,name):
    fig.savefig(FIG/f"{name}.pdf",bbox_inches="tight")
    fig.savefig(FIG/f"{name}.png",bbox_inches="tight")
    plt.close(fig)

def paired_panel(ax,items,title):
    for i,(key,label,color) in enumerate(items):
        values=rr[rr.comparison.eq(key)].sort_values("repeat").delta_ll.to_numpy()
        y=i+np.linspace(-.11,.11,len(values))
        ax.scatter(values,y,s=15,c=GREY,alpha=.8,zorder=2)
        ax.scatter(values.mean(),i,s=55,c=color,edgecolor="white",linewidth=.7,zorder=3)
    ax.axvline(0,color=".25",lw=.8,ls="--")
    ax.set_yticks(range(len(items)),[x[1] for x in items]);ax.invert_yaxis()
    ax.set_xlabel("Paired log-loss difference (positive favors addition)")
    ax.set_title(title,loc="left",fontweight="bold",pad=12)
    ax.grid(axis="x",alpha=.14);ax.set_axisbelow(True)

fig,axes=plt.subplots(1,2,figsize=(13,5.0),gridspec_kw={"width_ratios":[1.1,1]})
items=[("M2_vs_E4","Full E4: 9 features",BLUE),
       ("M2_vs_E4_only_perm_entropy","Permutation entropy only: 3",BLUE),
       ("M2_vs_E4_only_hjorth_mobility","Hjorth mobility only: 3",BLUE),
       ("M2_vs_E4_only_hjorth_complexity","Hjorth complexity only: 3",BLUE),
       ("M2_vs_E4_without_perm_entropy","E4 without entropy: 6",ORANGE),
       ("M2_vs_E4_without_hjorth_mobility","E4 without mobility: 6",ORANGE),
       ("M2_vs_E4_without_hjorth_complexity","E4 without complexity: 6",ORANGE)]
paired_panel(axes[0],items,"A   E4 components — all compared with M2")
paired_panel(axes[1],[("M2_vs_original_E0","Original E0: 12",GREY),("M2_vs_E0_relative9","Relative spectrum: 9",GREEN),
                     ("M2_vs_E0_PCA9","Spectral PCA: 9",GREEN),("M2_vs_E4","Full E4: 9",BLUE)],
             "B   Matched feature counts — compared with M2")
fig.suptitle("Post-hoc representation comparisons",fontsize=13,fontweight="bold")
fig.text(.5,.015,"Large points: means. Small points: 10 correlated CV repeats; they are not independent replications or confidence intervals.",ha="center",fontsize=8)
fig.tight_layout(rect=[0,.05,1,.95],w_pad=3)
save(fig,"PH1_components_and_dimension")

fig,axes=plt.subplots(1,2,figsize=(12.8,4.8))
paired_panel(axes[0],[("M2_vs_E4","E4 | M2 baseline",BLUE),("M2_D_vs_E4_D","E4 | M2 + device baseline",BLUE),
                     ("M2_DQ_vs_E4_DQ","E4 | M2 + device + quality baseline",BLUE),
                     ("M2_vs_original_adaptive","Adaptive EEG | M2 baseline",ORANGE),
                     ("M2_D_vs_adaptive_D","Adaptive EEG | M2 + device",ORANGE),
                     ("M2_DQ_vs_adaptive_DQ","Adaptive EEG | M2 + device + quality",ORANGE)],
             "A   Matched-baseline sensitivities")
paired_panel(axes[1],[("M2_vs_original_adaptive","Mandatory EEG policy",ORANGE),("M2_vs_optout","EEG opt-out policy",ORANGE)],
             "B   Policy comparison — M2 baseline")
choices=pd.read_csv(RESULT/"optout_choices.csv").representation.value_counts()
axes[1].text(.02,.47,"Opt-out selections across 50 outer fits:\n"+", ".join(f"{k}: {choices.get(k,0)}" for k in ["M2","E0","E1","E2"]),transform=axes[1].transAxes,fontsize=9,
             bbox=dict(boxstyle="round,pad=.5",facecolor="#F0F3F5",edgecolor="none"))
fig.suptitle("Post-hoc acquisition and model-selection checks",fontsize=13,fontweight="bold")
fig.text(.5,.012,"Device and quality variables enter both models in each paired comparison. Quality = clean-window fraction and interpolation count.",ha="center",fontsize=8)
fig.tight_layout(rect=[0,.06,1,.95],w_pad=3)
save(fig,"PH2_device_quality_and_policy")

participants=pd.read_csv(RESULT/"participant_loss_diagnostics.csv")
fig,axes=plt.subplots(1,2,figsize=(11.4,4.3))
for ax,name,label,color in zip(axes,["original_adaptive","E4"],["A   Original adaptive EEG","B   E4 temporal descriptors"],[ORANGE,BLUE]):
    values=np.sort(participants[participants.model.eq(name)].mean_delta_ll.to_numpy())
    ax.scatter(range(1,97),values,c=np.where(values>=0,color,GREY),s=15)
    ax.axhline(0,c=".3",lw=.8,ls="--");ax.axhline(values.mean(),c=color,lw=1)
    ax.set_title(label,loc="left",fontweight="bold")
    ax.set_xlabel("Participant rank within this panel (independently sorted)")
    ax.set_ylabel("Mean paired log-loss difference across repeats")
    ax.text(.03,.95,f"Positive participant means: {(values>0).sum()}/96\nOverall mean: {values.mean():+.4f}",transform=ax.transAxes,va="top",fontsize=9)
    ax.grid(axis="y",alpha=.15)
fig.suptitle("Post-hoc distribution of held-out gains and losses",fontweight="bold",fontsize=13)
fig.text(.5,.01,"One point per participant; all 10 held-out losses are averaged within participant. Positive values favor EEG. No participants excluded.",ha="center",fontsize=8)
fig.tight_layout(rect=[0,.06,1,.93])
save(fig,"PH3_participant_gains_losses")

coef=pd.read_csv(RESULT/"coefficient_summary.csv")
names=[f"{r}_{q}" for q in ["perm_entropy","hjorth_mobility","hjorth_complexity"] for r in ["frontal","temporal","posterior"]]
labels=[n.replace("perm_entropy","entropy").replace("hjorth_","").replace("_"," ").title() for n in names]
c=coef[coef.model.eq("E4")].set_index("feature").loc[names]
fig,ax=plt.subplots(figsize=(7.7,5.2))
y=np.arange(9);ax.hlines(y,c["min"],c["max"],color=GREY,lw=2);ax.scatter(c["median"],y,s=38,color=BLUE,zorder=3)
ax.axvline(0,c=".3",lw=.8,ls="--");ax.set_yticks(y,labels);ax.invert_yaxis()
ax.set_xlabel("Standardized logistic coefficient (log-odds scale)")
ax.set_title("Post-hoc E4 coefficient stability",loc="left",fontweight="bold")
fig.text(.5,.015,"Dots: medians across 50 correlated outer fits. Lines: minimum–maximum ranges, not confidence intervals.\nCoefficients describe the fitted predictive model and are not causal or source-localized effects.",ha="center",fontsize=8)
fig.tight_layout(rect=[0,.1,1,1]);save(fig,"PH4_coefficient_stability")

signal_summary=HERE/"signal_controls/comparison_summary.csv"
if signal_summary.exists():
    s=pd.read_csv(signal_summary);v=s[s.baseline.eq("M2")].delta_ll.to_numpy()
    fig,axes=plt.subplots(1,2,figsize=(11.4,4.4))
    for ax,metric,label in zip(axes,["delta_ll","delta_auroc"],["Paired log-loss improvement","Paired AUROC improvement"]):
        v=s[s.baseline.eq("M2")][metric].to_numpy();observed=float(ss.loc["M2_vs_E4",metric])
        ax.scatter(np.arange(1,21),v,c=GREY,s=30,label="Spectrum-preserving phase controls")
        ax.axhline(observed,c=BLUE,lw=1.5,label="Observed E4")
        ax.axhline(0,c=".3",lw=.8,ls="--")
        ax.set_xlabel("Surrogate dataset index");ax.set_ylabel(label);ax.set_xticks([1,5,10,15,20]);ax.grid(axis="y",alpha=.15)
    axes[0].legend(loc="best",frameon=False,fontsize=8)
    fig.suptitle("Post-hoc Fourier-phase stress test of E4",fontweight="bold",fontsize=13)
    fig.text(.5,.01,"Each point averages the same 10 nested-CV repeats. Controls preserve window-wise spectra, not all signal properties.\nSurrogate datasets and CV repeats are not independent samples; no significance test is shown.",ha="center",fontsize=8)
    fig.tight_layout(rect=[0,.09,1,.94]);save(fig,"PH5_spectrum_preserving_controls")
print("Figures generated:",", ".join(p.name for p in FIG.glob("*.png")))
