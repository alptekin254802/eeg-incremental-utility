"""Verify and summarize every prespecified comparison in the exploratory addendum."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from controls import loss,metrics,sha,dump,csv
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--release',type=Path,required=True);ap.add_argument('--transfer',type=Path,required=True);a=ap.parse_args();w=a.output;r=a.release;out=w/'summary';out.mkdir(exist_ok=False)
    lock=json.loads((w/'audit/PLAN_LOCK.json').read_text());assert sha(w/'ANALYSIS_PLAN.txt')==lock['plan_sha256']
    for part in ['age','bandwidth','extract','offset']:assert json.loads((w/part/'COMPLETE.json').read_text())['status']=='PASS'
    src=json.loads((w/'audit/INPUT_IDENTITIES.json').read_text())
    for label,root in [('release',r),('transfer',a.transfer)]:
        for n,h in src[label].items():assert sha(root/n)==h,(label,n)
    old=pd.read_csv(r/'expected/posthoc/results/outer_predictions.csv',float_precision='round_trip')
    age=pd.read_csv(w/'age/PREDICTIONS.csv',float_precision='round_trip');band=pd.read_csv(w/'bandwidth/PREDICTIONS.csv',float_precision='round_trip')
    keys=['repeat','outer_fold','participant_id','y'];checks={}
    for replay,ref in [('replay_baseline','M2'),('replay_entropy','E4'),('replay_baseline_DQ','M2_DQ')]:
        z=age[age.model==replay].merge(old[old.model==ref],on=keys,suffixes=('_new','_ref'),validate='one_to_one');err=float(np.max(abs(z.p_new-z.p_ref)));assert len(z)==20 and err<1e-7,(replay,err);checks[replay]=err
    for part,frame in [('age',age),('bandwidth',band)]:
        tuning=pd.read_csv(w/part/'TUNING.csv');assert len(tuning.groupby(['model','repeat','outer_fold']))*5==len(tuning)
        for (_,_,_),g in tuning.groupby(['model','repeat','outer_fold']):
            expected=min(g.to_dict('records'),key=lambda z:(round(z['inner_logloss'],12),z['C']));actual=g[g.selected].iloc[0];assert len(g[g.selected])==1 and actual.C==expected['C']
        for m,g in frame[~frame.model.str.startswith('replay')].groupby('model'):
            assert len(g)==960 and not g.duplicated(['repeat','participant_id']).any()
            z=g[keys].merge(old[old.model=='M2'][keys],on=keys,validate='one_to_one');assert len(z)==960
    tuning=pd.read_csv(w/'age/TUNING.csv');choices=pd.read_csv(w/'age/CHOICES.csv');selected=tuning[tuning.selected & tuning.model.isin(['E0_spline','E1_spline','E2_spline'])]
    for row in choices.to_dict('records'):
        g=selected[(selected.repeat==row['repeat'])&(selected.outer_fold==row['outer_fold'])];names=['E0_spline','E1_spline','E2_spline'];best=min(g.to_dict('records'),key=lambda z:(round(z['inner_logloss'],12),names.index(z['model']),z['C']));assert row['selected']==best['model']
        p1=age[(age.model=='selector_spline')&(age.repeat==row['repeat'])&(age.outer_fold==row['outer_fold'])].set_index('participant_id').p
        p2=age[(age.model==row['selected'])&(age.repeat==row['repeat'])&(age.outer_fold==row['outer_fold'])].set_index('participant_id').p
        assert np.array_equal(p1.sort_index(),p2.sort_index())
    cases=[('age_primary',age,'baseline_spline','selector_spline','M2','original_adaptive'),('age_entropy',age,'baseline_spline','entropy_spline','M2','E4'),('age_entropy_DQ',age,'baseline_spline_DQ','entropy_spline_DQ','M2_DQ','E4_DQ'),('bandwidth_entropy',pd.concat([band,old[old.model=='M2']]),'M2','entropy_lowpass30','M2','E4'),('bandwidth_entropy_DQ',pd.concat([band,old[old.model=='M2_DQ']]),'M2_DQ','entropy_lowpass30_DQ','M2_DQ','E4_DQ')]
    rng=np.random.default_rng(2026110302);draw=rng.integers(0,96,size=(5000,96));summ=[];reps=[];persons=[];drawrows=[]
    for name,frame,base,aug,refbase,refaug in cases:
        def paired(f,b,e):
            z=f[f.model==b][keys+['p']].merge(f[f.model==e][keys+['p']],on=keys,suffixes=('_base','_add'),validate='one_to_one');assert len(z)==960
            z['delta']=loss(z.y,z.p_base)-loss(z.y,z.p_add);return z
        z=paired(frame,base,aug);ref=paired(old,refbase,refaug);z=z.merge(ref[keys+['delta']],on=keys,suffixes=('','_ref'),validate='one_to_one')
        for repeat,g in z.groupby('repeat'):
            m0=metrics(g.y,g.p_base.to_numpy());m1=metrics(g.y,g.p_add.to_numpy());reps.append(dict(comparison=name,repeat=int(repeat),baseline_logloss=m0['logloss'],augmented_logloss=m1['logloss'],delta_ll=m0['logloss']-m1['logloss'],reference_delta_ll=float(g.delta_ref.mean()),delta_auroc=m1['auroc']-m0['auroc'],delta_brier=m0['brier']-m1['brier']))
        d=z.groupby('participant_id')[['delta','delta_ref']].mean().sort_index();assert len(d)==96
        change=d.delta-d.delta_ref;dv=d.delta.to_numpy()[draw].mean(axis=1);cv=change.to_numpy()[draw].mean(axis=1);q=np.quantile(dv,[.025,.975]);qc=np.quantile(cv,[.025,.975]);rr=pd.DataFrame([x for x in reps if x['comparison']==name]);iqr=np.quantile(rr.delta_ll,[.25,.75])
        summ.append(dict(comparison=name,baseline=base,augmented=aug,baseline_logloss=float(rr.baseline_logloss.mean()),augmented_logloss=float(rr.augmented_logloss.mean()),delta_ll=float(rr.delta_ll.mean()),reference_delta_ll=float(rr.reference_delta_ll.mean()),change_from_reference=float(change.mean()),lower=q[0],upper=q[1],change_lower=qc[0],change_upper=qc[1],delta_auroc=float(rr.delta_auroc.mean()),delta_brier=float(rr.delta_brier.mean()),positive_repeats=int((rr.delta_ll>0).sum()),median=float(rr.delta_ll.median()),iqr_low=iqr[0],iqr_high=iqr[1],minimum=float(rr.delta_ll.min()),maximum=float(rr.delta_ll.max())))
        for pid,row in d.iterrows():persons.append(dict(comparison=name,participant_id=pid,delta_ll=row.delta,reference_delta_ll=row.delta_ref,change=row.delta-row.delta_ref))
        for i,(v,c) in enumerate(zip(dv,cv)):drawrows.append(dict(comparison=name,draw=i,delta_ll=v,change_from_reference=c))
    csv(out/'SENSITIVITY_SUMMARY.csv',summ);csv(out/'REPEAT_RESULTS.csv',reps);csv(out/'PARTICIPANT_DIFFERENCES.csv',persons);csv(out/'CONDITIONAL_DRAWS.csv',drawrows)
    off=pd.read_csv(w/'offset/CONTRASTS.csv',float_precision='round_trip');arch=pd.read_csv(a.transfer/'analysis/results/PRINCIPAL_CONTRASTS.csv',float_precision='round_trip');oo=off[off.correction=='original'].merge(arch,on=['direction','model'],suffixes=('_new','_ref'));assert len(oo)==4
    for col in ['delta_ll','equal_class_delta_ll']:assert np.max(abs(oo[col+'_new']-oo[col+'_ref']))<1e-12
    for _,g in off.groupby(['direction','model']):
        assert np.ptp(g.baseline_auroc)<1e-12 and np.ptp(g.augmented_auroc)<1e-12
    raw=json.loads((w/'extract/RAW_RECONSTRUCTION.json').read_text());assert len(raw)==96 and max(z['e4_max_error'] for z in raw)<1e-8
    checks.update(plan_unchanged=True,original_input_files_unchanged=True,all_50_nested_partitions_verified=True,all_tuning_and_selector_choices_verified=True,raw_participants=96,max_raw_e0_error=max(z['e0_max_error'] for z in raw),max_raw_e4_error=max(z['e4_max_error'] for z in raw),offset_original_scores_reproduced=True,offset_auroc_invariant=True,new_fit_comparisons=5,offset_comparisons=4,new_outer_models=450,reference_outer_models_replayed=3,scope='Complete planned local sensitivity fits and raw-derived bandwidth features; not a rerun of the entire historical project or an external validation.')
    dump(out/'VERIFICATION.json',dict(status='PASS',**checks));print(pd.DataFrame(summ).to_string(index=False))
if __name__=='__main__':main()
