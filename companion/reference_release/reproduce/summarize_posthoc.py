"""Validate saved post-hoc outputs and produce a complete reviewable handoff."""
import sys
sys.dont_write_bytecode=True
import json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import spearmanr
import run_posthoc as core

HERE=core.HERE;R=core.RESULT;S=HERE/"signal_controls"

def table(frame,columns=None):
    if columns is not None:frame=frame[columns]
    header="| "+" | ".join(str(c) for c in frame.columns)+" |\n"
    header+="| "+" | ".join("---" for _ in frame.columns)+" |\n"
    for row in frame.itertuples(index=False,name=None):
        vals=[f"{x:.6f}" if isinstance(x,(float,np.floating)) else str(x) for x in row]
        header+="| "+" | ".join(vals)+" |\n"
    return header

def main():
    assert (R/"RUN_COMPLETE.json").exists() and (S/"RUN_COMPLETE.json").exists()
    core.freeze_sources()
    p=pd.read_csv(R/"outer_predictions.csv");s=pd.read_csv(R/"comparison_summary.csv").set_index("comparison")
    repeats=pd.read_csv(R/"repeat_results.csv")
    for name,g in p.groupby("model"):
        assert len(g)==960 and g.participant_id.nunique()==96 and not g.duplicated(["repeat","participant_id"]).any()
        assert np.isfinite(g.p).all() and g.p.between(0,1).all()
        assert g.groupby("repeat").size().eq(96).all()
    for _,r in s.reset_index().iterrows():
        g=repeats[repeats.comparison.eq(r.comparison)]
        assert len(g)==10 and abs(g.delta_ll.mean()-r.delta_ll)<1e-12
    opt=pd.read_csv(R/"optout_choices.csv")
    old_inner=pd.read_csv(core.AUDIT/"preregistered_analysis/PRIMARY_INNER_SELECTION.csv")
    selected=old_inner[old_inner.model_family.isin(["M2_SELECTED","E0_SELECTED","E1_SELECTED","E2_SELECTED"])]
    assert selected["rank"].eq(1).all() and selected.fallback_rank.isna().all()
    assert len(opt)==50
    # Independent identity: paired contribution terms must reconstruct the actual held-out logit change.
    change=pd.read_csv(R/"paired_logit_change_decomposition.csv")
    key=["repeat","outer_fold","participant_id"]
    pairs=p[p.model.eq("M2")][key+["p"]].merge(p[p.model.eq("E4")][key+["p"]],on=key,suffixes=("_m2","_e4"),validate="one_to_one")
    pairs=pairs.merge(change,on=key,validate="one_to_one")
    terms=["baseline_refit_logit_delta","perm_entropy_logit","hjorth_mobility_logit","hjorth_complexity_logit"]
    error=float(np.max(np.abs((logit(pairs.p_e4)-logit(pairs.p_m2))-pairs[terms].sum(axis=1))))
    assert error<1e-10,error
    # Contribution aggregation respects participant as the unit.
    c=change.groupby("participant_id")[terms].agg(lambda x:float(np.mean(np.abs(x)))).reset_index()
    c.to_csv(R/"participant_absolute_logit_terms.csv",index=False,float_format="%.17g")
    c[terms].agg(["mean","median","min","max"]).rename_axis("summary").reset_index().to_csv(R/"logit_term_summary.csv",index=False,float_format="%.17g")
    part=pd.read_csv(R/"participant_loss_diagnostics.csv")
    prows=[]
    for name,g in part.groupby("model"):
        values=g.mean_delta_ll.to_numpy()
        prows.append(dict(model=name,n=len(g),mean_delta=float(values.mean()),median_delta=float(np.median(values)),positive_n=int((values>0).sum()),
             positive_contribution_to_mean=float(values[values>0].sum()/96),negative_contribution_to_mean=float(values[values<0].sum()/96),
             top5_positive_contribution_to_mean=float(np.sort(values[values>0])[-5:].sum()/96),
             top5_negative_contribution_to_mean=float(np.sort(values[values<0])[:5].sum()/96)))
    part_summary=pd.DataFrame(prows);part_summary.to_csv(R/"participant_diagnostic_summary.csv",index=False,float_format="%.17g")
    roster=pd.read_csv(R/"roster_device_quality.csv").set_index("participant_id")
    f=pd.read_csv(core.AUDIT/"exploratory_eeg_battery/E4_RAW_FEATURES.csv").set_index("participant_id").loc[roster.index]
    nuisance=pd.DataFrame({"device_epocplus":roster.device.eq("Epoc+").astype(float),"clean_window_fraction":roster.clean_window_fraction,"interpolation_count":roster.interpolation_count})
    cor=[]
    for feature in f:
        for q in nuisance:cor.append(dict(feature=feature,covariate=q,spearman=float(spearmanr(f[feature],nuisance[q]).statistic)))
    pd.DataFrame(cor).to_csv(R/"e4_device_quality_correlations.csv",index=False,float_format="%.17g")
    roster.groupby(["device","y"])[["clean_window_fraction","interpolation_count"]].agg(["count","mean","median","min","max"]).to_csv(R/"device_group_quality_distributions.csv",float_format="%.17g")
    sig=pd.read_csv(S/"comparison_summary.csv")
    sp=pd.read_csv(S/"outer_predictions.csv")
    assert len(sp)==19200 and sp.surrogate.nunique()==20
    assert not sp.duplicated(["surrogate","repeat","participant_id"]).any()
    assert sp.groupby(["surrogate","repeat"]).size().eq(96).all()
    checks=pd.read_csv(S/"reconstruction_checks.csv")
    assert len(checks)==96
    for col in ("e0_error","e4_error","vectorized_error","spectrum_relative_error"):assert checks[col].max()<1e-10
    control=sig[sig.baseline.eq("M2")]
    observed=float(s.loc["M2_vs_E4","delta_ll"])
    sigstats=dict(n_surrogate_datasets=20,mean_delta_ll=float(control.delta_ll.mean()),min_delta_ll=float(control.delta_ll.min()),max_delta_ll=float(control.delta_ll.max()),
                  observed_delta_ll=observed,observed_minus_surrogate_mean=observed-float(control.delta_ll.mean()),surrogates_above_observed=int((control.delta_ll>observed).sum()),
                  mean_delta_auroc=float(control.delta_auroc.mean()),mean_delta_brier=float(control.delta_brier.mean()))
    core.write_json(S/"descriptive_summary.json",sigstats)
    core.write_json(HERE/"FINAL_IMPLEMENTATION_CHECKS.json",dict(completed_utc=datetime.now(timezone.utc).isoformat(),
         scope="Post-hoc implementation checks; not a formal publication audit",source_manifest_unchanged=True,models=int(p.model.nunique()),comparisons=len(s),
         all_participant_repeat_rows_unique=True,original_optout_refits_without_fallback=True,logit_decomposition_max_error=error,
         surrogate_predictions=len(sp),raw_reconstructions=96,surrogate_reconstruction_max_errors=checks[["e0_error","e4_error","vectorized_error","spectrum_relative_error"]].max().to_dict()))

if __name__=="__main__": main()
