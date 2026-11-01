"""Post-hoc explanatory analyses with training-only model selection."""
from __future__ import annotations
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import importlib.util
import json
import platform
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(os.environ["EEG_RELEASE_ROOT"])
HERE = Path(os.environ["EEG_RUN_ROOT"]) / "posthoc"
HERE.mkdir(parents=True, exist_ok=True)
AUDIT = ROOT / "audit"
RESULT = HERE / "results"
GRID = [0.01, 0.1, 1., 10., 100.]
def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
PRIMARY = load_module(AUDIT / "preregistered_analysis/run_preregistered_analysis.py", "frozen_primary")
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
def csv(frame, name):
    frame.to_csv(RESULT / name, index=False, float_format="%.17g")
def loss(y, p):
    p = np.clip(np.asarray(p, float), 1e-8, 1-1e-8)
    return -np.asarray(y)*np.log(p)-(1-np.asarray(y))*np.log1p(-p)
def metrics(y, p):
    return dict(logloss=float(loss(y,p).mean()), auroc=float(roc_auc_score(y,p)), brier=float(np.mean((y-p)**2)))

def load_data():
    base, measurement = PRIMARY.load_measurements()
    ids = base.participant_id.tolist()
    old = pd.read_csv(AUDIT / "preregistered_analysis/PRIMARY_OUTER_PREDICTIONS.csv")
    old = old[old.outcome.eq("group")].copy()
    assert len(old)==960 and not old.duplicated(["repeat","participant_id"]).any()
    y = old[old.repeat.eq(0)].set_index("participant_id").loc[ids,"y"].to_numpy(int)
    folds = PRIMARY.build_folds(ids, y, "group")
    manifest = pd.read_csv(AUDIT / "preregistered_analysis/PRIMARY_FOLD_MANIFEST.csv")
    # Verify both nested levels against frozen membership, not just RNG seeds.
    for outer in folds["outer"]:
        records = manifest[(manifest.repeat==outer["repeat"]) & (manifest.outer_fold==outer["outer_fold"])]
        for role,key in [("train","outer_train"),("test","outer_test")]:
            assert set(records[(records.level=="outer") & (records.role==role)].participant_id)=={ids[i] for i in outer[key]}
        for inner in outer["inner_folds"]:
            for role,key in [("train","inner_train"),("validation","inner_valid")]:
                subset=records[(records.level=="inner") & (records.inner_fold==inner["inner_fold"])]
                actual_role = role if role in set(subset.role) else ("valid" if role=="validation" else role)
                assert set(subset[subset.role==actual_role].participant_id)=={ids[i] for i in inner[key]}, (set(subset.role), role)
            assert not set(inner["inner_train"]) & set(inner["inner_valid"])
            assert not set(outer["outer_test"]) & set(inner["inner_train"]) | set(outer["outer_test"]) & set(inner["inner_valid"])
        pids=set(old[(old.repeat==outer["repeat"]) & (old.outer_fold==outer["outer_fold"])].participant_id)
        assert pids=={ids[i] for i in outer["outer_test"]}
    e4=pd.read_csv(AUDIT / "exploratory_eeg_battery/E4_RAW_FEATURES.csv").set_index("participant_id").loc[ids]
    qc=pd.read_csv(AUDIT / "stage1_blind_feature_extraction/EEG_PREPROCESSING_QC.csv").set_index("participant_id").loc[ids]
    assert set(qc.device)=={"Epoc X","Epoc+"}
    nuisance=np.column_stack([qc.device.eq("Epoc+").astype(float),qc.clean_window_fraction,qc.interpolation_count])
    assert np.isfinite(nuisance).all() and np.isfinite(e4.to_numpy()).all()
    return dict(ids=ids,y=y,folds=folds,raw=PRIMARY.prepare_raw(base,measurement),e4=e4.to_numpy(float),
                e4_names=e4.columns.tolist(),nuisance=nuisance,old=old,qc=qc,base=base)

DATA=None
def initialize(data):
    global DATA
    DATA=data
    threadpool_limits(1)

def fit(x,y,c):
    for limit in (1000,10000):
        model=LogisticRegression(C=c,solver="lbfgs",tol=1e-6,max_iter=limit,fit_intercept=True,class_weight=None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(x,y)
        convergence=any(issubclass(w.category,ConvergenceWarning) for w in caught)
        if convergence and limit==1000: continue
        if convergence: raise RuntimeError("Convergence failure after retry")
        assert np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all()
        return model, limit==10000
    raise RuntimeError("No fit")

def variants():
    specs=[("M2", "none", 0), ("E4", "E4", 0)]
    for quantity in ("perm_entropy","hjorth_mobility","hjorth_complexity"):
        specs.extend([(f"E4_without_{quantity}",f"without:{quantity}",0),
                      (f"E4_only_{quantity}",f"only:{quantity}",0)])
    specs.extend([("E0_relative9","relative9",0),("E0_PCA9","pca9",0)])
    for suffix,width in [("D",1),("DQ",3)]:
        specs.extend([(f"M2_{suffix}","none",width),(f"E4_{suffix}","E4",width)])
        specs.extend([(f"{rep}_{suffix}",rep,width) for rep in ("E0","E1","E2")])
    return specs

def design(data,tr,te,kind,width,cache):
    raw=data["raw"]
    names=list(PRIMARY.M2_COLUMNS)
    if kind in ("E0","E1","E2"):
        if kind not in cache:
            a,b,_=PRIMARY.representation_features(raw,tr,te,kind)
            cache[kind]=(a,b)
        a,b=cache[kind]
        names += [f"{kind}_{i}" for i in range(a.shape[1]-7)]
    else:
        a,b=raw["m2"][tr],raw["m2"][te]
        if kind=="E4" or ":" in kind:
            cols=np.arange(9)
            if ":" in kind:
                action,q=kind.split(":")
                mask=np.array([n.endswith(q) for n in data["e4_names"]])
                cols=cols[mask if action=="only" else ~mask]
            a=np.column_stack([a,data["e4"][tr][:,cols]])
            b=np.column_stack([b,data["e4"][te][:,cols]])
            names += [data["e4_names"][i] for i in cols]
        elif kind=="relative9":
            cols=[i for i,n in enumerate(PRIMARY.E0_COLUMNS) if "rel_" in n]
            a=np.column_stack([a,raw["e0"][tr][:,cols]])
            b=np.column_stack([b,raw["e0"][te][:,cols]])
            names += [PRIMARY.E0_COLUMNS[i] for i in cols]
        elif kind=="pca9":
            sc=StandardScaler().fit(raw["e0"][tr])
            pca=PCA(n_components=9,svd_solver="full").fit(sc.transform(raw["e0"][tr]))
            assert np.linalg.matrix_rank(sc.transform(raw["e0"][tr]))>=9
            a=np.column_stack([a,pca.transform(sc.transform(raw["e0"][tr]))])
            b=np.column_stack([b,pca.transform(sc.transform(raw["e0"][te]))])
            names += [f"E0_PC{i+1}" for i in range(9)]
        elif kind!="none": raise ValueError(kind)
    if width:
        a=np.column_stack([a,data["nuisance"][tr,:width]])
        b=np.column_stack([b,data["nuisance"][te,:width]])
        names += ["device_epocplus","clean_window_fraction","interpolation_count"][:width]
    sc=StandardScaler().fit(a)
    return sc.transform(a),sc.transform(b),names

def outer_job(outer):
    data=DATA
    pred,selection,coefficients,contributions=[],[],[],[]
    rep,fold=outer["repeat"],outer["outer_fold"]
    inner_cache=[{} for _ in outer["inner_folds"]]
    out_cache={}
    retries=0
    for name,kind,width in variants():
        scores={}
        mats=[design(data,i["inner_train"],i["inner_valid"],kind,width,cache)
              for i,cache in zip(outer["inner_folds"],inner_cache)]
        for c in GRID:
            losses=[]
            for inner,(x,v,_) in zip(outer["inner_folds"],mats):
                model,retry=fit(x,data["y"][inner["inner_train"]],c)
                retries += int(retry)
                losses.extend(loss(data["y"][inner["inner_valid"]],model.predict_proba(v)[:,1]).tolist())
            scores[c]=float(np.mean(losses))
        best=min(GRID,key=lambda c:(round(scores[c],12),c))
        for c in GRID:
            selection.append(dict(model=name,repeat=rep,outer_fold=fold,C=c,inner_logloss=scores[c],selected=c==best))
        x,v,names=design(data,outer["outer_train"],outer["outer_test"],kind,width,out_cache)
        model,retry=fit(x,data["y"][outer["outer_train"]],best)
        retries += int(retry)
        probability=model.predict_proba(v)[:,1]
        for j,idx in enumerate(outer["outer_test"]):
            pred.append(dict(model=name,repeat=rep,outer_fold=fold,participant_id=data["ids"][idx],
                             y=int(data["y"][idx]),p=float(probability[j]),selected_C=best))
        if name in ("M2","E4"):
            for n,c in zip(names,model.coef_[0]):
                coefficients.append(dict(model=name,repeat=rep,outer_fold=fold,feature=n,coefficient=float(c),selected_C=best))
            terms=v*model.coef_[0]
            assert np.max(np.abs(expit(model.intercept_[0]+terms.sum(axis=1))-probability))<1e-12
            for j,idx in enumerate(outer["outer_test"]):
                row=dict(model=name,repeat=rep,outer_fold=fold,participant_id=data["ids"][idx],intercept=float(model.intercept_[0]))
                row.update({n:float(terms[j,k]) for k,n in enumerate(names)})
                contributions.append(row)
    return pred,selection,coefficients,contributions,retries

def summarize_predictions(pred, comparisons):
    rows=[]
    for label,baseline,augmented in comparisons:
        a=pred[pred.model.eq(baseline)]
        b=pred[pred.model.eq(augmented)]
        merged=a.merge(b,on=["repeat","outer_fold","participant_id","y"],suffixes=("_base","_add"),validate="one_to_one")
        assert len(merged)==960,(label,len(merged))
        for repeat,sub in merged.groupby("repeat"):
            m0,m1=metrics(sub.y.to_numpy(),sub.p_base.to_numpy()),metrics(sub.y.to_numpy(),sub.p_add.to_numpy())
            rows.append(dict(comparison=label,baseline=baseline,augmented=augmented,repeat=int(repeat),n=len(sub),
                             baseline_logloss=m0["logloss"],augmented_logloss=m1["logloss"],
                             delta_ll=m0["logloss"]-m1["logloss"],delta_auroc=m1["auroc"]-m0["auroc"],
                             delta_brier=m0["brier"]-m1["brier"]))
    repeats=pd.DataFrame(rows)
    summary=repeats.groupby(["comparison","baseline","augmented"],sort=False).agg(
        baseline_logloss=("baseline_logloss","mean"),augmented_logloss=("augmented_logloss","mean"),
        delta_ll=("delta_ll","mean"),delta_auroc=("delta_auroc","mean"),delta_brier=("delta_brier","mean"),
        positive_repeats=("delta_ll",lambda x:int((x>0).sum())),min_repeat_delta=("delta_ll","min"),max_repeat_delta=("delta_ll","max")).reset_index()
    return repeats,summary

def postprocess(data,pred,sel,coef,contrib,retries):
    validations=dict(n=96,outer_fits=50,nested_membership_verified=True,convergence_retries=retries)
    old=data["old"]
    eold=pd.read_csv(AUDIT/"exploratory_eeg_battery/OUTER_PREDICTIONS.csv")
    eold=eold[eold.family.eq("E4")]
    for name,src,col in [("M2",old,"p_m2"),("E4",eold,"p_eeg")]:
        merged=pred[pred.model.eq(name)].merge(src,on=["repeat","outer_fold","participant_id","y"],validate="one_to_one")
        error=float(np.max(np.abs(merged.p-merged[col])))
        assert len(merged)==960 and error<1e-7,(name,error)
        validations[f"{name}_max_probability_error"]=error
    # Derive the two adjusted adaptive procedures strictly from inner scores.
    selected=sel[sel.selected].copy()
    adaptive_choices=[]
    for suffix in ("D","DQ"):
        for (rep,fold),sub in selected[selected.model.isin([f"{r}_{suffix}" for r in ("E0","E1","E2")])].groupby(["repeat","outer_fold"]):
            candidates=sub.to_dict("records")
            choice=min(candidates,key=lambda r:(round(r["inner_logloss"],12),["E0","E1","E2"].index(r["model"].split("_")[0]),r["C"]))
            picked=pred[(pred.model==choice["model"]) & (pred.repeat==rep) & (pred.outer_fold==fold)].copy()
            picked["model"]=f"adaptive_{suffix}"
            pred=pd.concat([pred,picked],ignore_index=True)
            adaptive_choices.append({**choice,"policy":f"adaptive_{suffix}"})
    archived_rows=[]
    for name,col in [("original_adaptive","p_adaptive_m3"),("original_E0","p_e0"),("original_E1","p_e1"),("original_E2","p_e2")]:
        a=old[["repeat","outer_fold","participant_id","y",col]].rename(columns={col:"p"}).copy()
        a["model"]=name
        archived_rows.append(a)
    pred=pd.concat([pred,*archived_rows],ignore_index=True)
    inner=pd.read_csv(AUDIT/"preregistered_analysis/PRIMARY_INNER_SELECTION.csv")
    inner=inner[inner.outcome.eq("group") & inner.model_family.isin(["M2","E0","E1","E2"])]
    choices=[]
    for (rep,fold),sub in inner.groupby(["repeat","outer_fold"]):
        assert len(sub)==20 and sub.valid.all()
        choice=min(sub.to_dict("records"),key=lambda r:(round(r["pooled_logloss"],12),["M2","E0","E1","E2"].index(r["representation"]),r["C"]))
        family=choice["representation"]
        src=old[(old.repeat==rep)&(old.outer_fold==fold)]
        cpref="m2" if family=="M2" else family.lower()
        assert src[f"{cpref}_C"].eq(choice["C"]).all()
        assert src[f"{cpref}_representation"].eq(family).all()
        pick=src[["repeat","outer_fold","participant_id","y",f"p_{cpref}"]].rename(columns={f"p_{cpref}":"p"}).copy()
        pick["model"]="optout"
        pick["selected_C"]=choice["C"]
        pred=pd.concat([pred,pick],ignore_index=True)
        choices.append(dict(repeat=int(rep),outer_fold=int(fold),representation=family,C=choice["C"],inner_logloss=choice["pooled_logloss"]))
    comparisons=[]
    for model in pred.model.unique():
        if model=="M2": continue
        baseline="M2_DQ" if model.endswith("_DQ") and model!="M2_DQ" else "M2_D" if model.endswith("_D") and model!="M2_D" else "M2"
        comparisons.append((f"{baseline}_vs_{model}",baseline,model))
    for model in [n for n,_,_ in variants() if n.startswith("E4_without_") or n.startswith("E4_only_")]+["E0_relative9","E0_PCA9"]:
        comparisons.append((f"E4_vs_{model}","E4",model))
    comparisons.append(("original_adaptive_vs_optout","original_adaptive","optout"))
    rr,ss=summarize_predictions(pred,comparisons)
    csv(pred,"outer_predictions.csv");csv(sel,"inner_selection.csv");csv(pd.DataFrame(adaptive_choices),"adjusted_adaptive_choices.csv")
    csv(pd.DataFrame(choices),"optout_choices.csv");csv(rr,"repeat_results.csv");csv(ss,"comparison_summary.csv")
    csv(coef,"outer_coefficients.csv");csv(contrib,"heldout_logit_contributions.csv")
    csv(coef.groupby(["model","feature"]).coefficient.agg(["mean","median","min","max",lambda x:float((x>0).mean())]).reset_index().rename(columns={"<lambda_0>":"positive_fit_fraction"}),"coefficient_summary.csv")
    feature_frame=pd.DataFrame(data["e4"],columns=data["e4_names"])
    csv(feature_frame.corr(method="spearman").rename_axis("feature").reset_index(),"e4_spearman.csv")
    roster=pd.DataFrame(dict(participant_id=data["ids"],y=data["y"],device=data["qc"].device.to_numpy(),clean_window_fraction=data["nuisance"][:,1],interpolation_count=data["nuisance"][:,2]))
    csv(roster,"roster_device_quality.csv")
    csv(pd.crosstab(roster.device,roster.y).rename(columns={0:"control",1:"study_group"}).reset_index(),"device_by_group.csv")
    parts=[]
    for name in ("original_adaptive","E4","optout","E4_D","E4_DQ"):
        base="M2_DQ" if name.endswith("_DQ") else "M2_D" if name.endswith("_D") else "M2"
        z=pred[pred.model.eq(base)].merge(pred[pred.model.eq(name)],on=["repeat","outer_fold","participant_id","y"],suffixes=("_base","_add"),validate="one_to_one")
        z["delta_ll"]=loss(z.y,z.p_base)-loss(z.y,z.p_add)
        p=z.groupby(["participant_id","y"]).agg(mean_delta_ll=("delta_ll","mean"),min_delta_ll=("delta_ll","min"),max_delta_ll=("delta_ll","max"),p_baseline=("p_base","mean"),p_augmented=("p_add","mean")).reset_index()
        p["model"]=name;p["baseline"]=base
        parts.append(p.merge(roster,on=["participant_id","y"],validate="one_to_one"))
    participants=pd.concat(parts,ignore_index=True)
    csv(participants,"participant_loss_diagnostics.csv")
    csv(participants.groupby(["model","device","y"]).agg(n=("participant_id","size"),mean_delta_ll=("mean_delta_ll","mean"),median_delta_ll=("mean_delta_ll","median"),positive_n=("mean_delta_ll",lambda x:int((x>0).sum()))).reset_index(),"device_group_loss_diagnostics.csv")
    # Exact difference in logits: changed non-EEG terms + new EEG terms.
    c0=contrib[contrib.model.eq("M2")].dropna(axis=1,how="all")
    c1=contrib[contrib.model.eq("E4")]
    cd=c1.merge(c0,on=["repeat","outer_fold","participant_id"],suffixes=("_e4","_m2"),validate="one_to_one")
    change=cd[["repeat","outer_fold","participant_id"]].copy()
    change["baseline_refit_logit_delta"]=cd.intercept_e4-cd.intercept_m2
    for col in PRIMARY.M2_COLUMNS: change["baseline_refit_logit_delta"]+=cd[f"{col}_e4"]-cd[f"{col}_m2"]
    for q in ("perm_entropy","hjorth_mobility","hjorth_complexity"):
        change[q+"_logit"]=cd[[n for n in data["e4_names"] if n.endswith(q)]].sum(axis=1)
    csv(change,"paired_logit_change_decomposition.csv")
    write_json(RESULT/"implementation_checks.json",validations)
    print(ss[["comparison","delta_ll","positive_repeats"]].to_string(index=False),flush=True)

def freeze_sources():
    manifest=json.loads((ROOT/"RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    hashes={name:digest(ROOT/name) for name in manifest["sha256"]}
    assert hashes==manifest["sha256"], "Release inputs changed"
    write_json(HERE/"SOURCE_MANIFEST.json",dict(source_sha256=hashes,
        python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__))


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--jobs",type=int,default=6);parser.add_argument("--preflight",action="store_true")
    args=parser.parse_args()
    RESULT.mkdir(exist_ok=True)
    freeze_sources()
    data=load_data()
    print("SOURCE/FOLD CHECKS OK: 96 participants, all 50 outer and 250 inner partitions matched.",flush=True)
    if args.preflight:return
    rows=[[],[],[],[]];retries=0
    with ProcessPoolExecutor(max_workers=args.jobs,initializer=initialize,initargs=(data,)) as pool:
        futures={pool.submit(outer_job,o):o for o in data["folds"]["outer"]}
        for count,future in enumerate(as_completed(futures),1):
            result=future.result()
            for i in range(4):rows[i].extend(result[i])
            retries+=result[4]
            print(f"Completed outer fit {count}/50",flush=True)
    frames=[pd.DataFrame(r).sort_values(["model","repeat","outer_fold"]) for r in rows]
    postprocess(data,*frames,retries)
    freeze_sources()
    write_json(RESULT/"RUN_COMPLETE.json",dict(completed_utc=datetime.now(timezone.utc).isoformat(),script_sha256=digest(__file__),release_manifest_sha256=digest(ROOT/"RELEASE_MANIFEST.json")))
if __name__=="__main__":main()
