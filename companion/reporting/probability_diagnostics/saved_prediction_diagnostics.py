"""Post-hoc diagnostics of saved probabilities; no predictive or calibration fits.

Requires Python >=3.10, NumPy and pandas. All input roots remain read-only.
Output paths must be outside the input roots and empty or not yet present.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd

PAIRS = [('unadjusted','M2','E4'), ('device','M2_D','E4_D'),
         ('device_quality','M2_DQ','E4_DQ')]
FOCAL = ['spectral_covariance_selector','entropy_hjorth']
TRANSFER_DRAWS = [0,1,17,999,4999]
ADULT_DRAWS = [0,1,17,999,9999]
QUANTILES = [.01,.05,.10,.25,.50,.75,.90,.95,.99]
TOL = 1e-10

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path, obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

def save(path, rows):
    pd.DataFrame(rows).to_csv(path,index=False,float_format='%.17g',lineterminator='\n')

def loss(y,p):
    p=np.clip(np.asarray(p,float),1e-8,1-1e-8)
    y=np.asarray(y,int)
    return -y*np.log(p)-(1-y)*np.log1p(-p)

def auc(y,p):
    y=np.asarray(y,int);p=np.asarray(p,float)
    d=p[y==1,None]-p[None,y==0]
    if not d.size:raise ValueError('AUROC requires both classes')
    return float(((d>0)+.5*(d==0)).mean())

def wrong_tail(y,p):
    y=np.asarray(y,int);p=np.asarray(p,float)
    return ((y==1)&(p<.1))|((y==0)&(p>.9))

def probability_summary(y,p):
    y=np.asarray(y,int);p=np.asarray(p,float);ll=loss(y,p);tail=wrong_tail(y,p)
    if not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all():
        raise ValueError('Nonfinite or out-of-range probabilities')
    return dict(n_rows=len(y),observed_prevalence=float(y.mean()),mean_p=float(p.mean()),
                p_min=float(p.min()),p_max=float(p.max()),
                **{f'p_q{round(q*100):02}':float(np.quantile(p,q,method='linear')) for q in QUANTILES},
                logloss=float(ll.mean()),brier=float(np.mean((y-p)**2)),auroc=auc(y,p),
                tail_wrong_y1_p_lt_01=int(((y==1)&(p<.1)).sum()),
                tail_wrong_y0_p_gt_09=int(((y==0)&(p>.9)).sum()),
                tail_wrong_total=int(tail.sum()),
                tail_own_loss_contribution=float(ll[tail].sum()/len(y)))

def paired_summary(y,p0,p1):
    y=np.asarray(y,int);p0=np.asarray(p0,float);p1=np.asarray(p1,float)
    d=loss(y,p0)-loss(y,p1);a,b=wrong_tail(y,p0),wrong_tail(y,p1)
    row=dict(n_rows=len(y),observed_prevalence=float(y.mean()),baseline_mean_p=float(p0.mean()),
             augmented_mean_p=float(p1.mean()),baseline_logloss=float(loss(y,p0).mean()),
             augmented_logloss=float(loss(y,p1).mean()),delta_ll=float(d.mean()),
             delta_brier=float(((y-p0)**2-(y-p1)**2).mean()),delta_auroc=auc(y,p1)-auc(y,p0),
             control_delta_ll=float(d[y==0].mean()),adhd_delta_ll=float(d[y==1].mean()),
             equal_class_delta_ll=float((d[y==0].mean()+d[y==1].mean())/2))
    masks={'baseline_only':a&~b,'augmented_only':~a&b,'both':a&b,'neither':~a&~b}
    for name,mask in masks.items():
        row[f'tail_{name}_n']=int(mask.sum())
        row[f'tail_{name}_delta_ll_contribution']=float(d[mask].sum()/len(y))
    row['tail_union_n']=int((a|b).sum())
    row['tail_union_delta_ll_contribution']=float(d[a|b].sum()/len(y))
    assert sum(row[f'tail_{k}_n'] for k in masks)==len(y)
    assert abs(sum(row[f'tail_{k}_delta_ll_contribution'] for k in masks)-row['delta_ll'])<1e-12
    return row

def paired_frame(pred,base,aug,keys):
    a=pred[pred.model==base][keys+['y','p']]
    b=pred[pred.model==aug][keys+['y','p']]
    z=a.merge(b,on=keys,validate='one_to_one',suffixes=('_base','_add')).sort_values(keys)
    if len(z)!=len(a) or len(z)!=len(b) or not z.y_base.equals(z.y_add):
        raise ValueError('Pair count or label mismatch')
    return z

def run(args):
    roots={k:Path(getattr(args,k)).resolve() for k in ['release_root','transfer_root','extension_root']}
    out=Path(args.output).resolve()
    for p in roots.values():
        if not p.is_dir():raise ValueError('An input root is not a directory')
        if out==p or p in out.parents:raise ValueError('Output cannot lie inside an input root')
    if out.exists() and any(out.iterdir()):raise ValueError('Output directory must be empty')
    out.mkdir(parents=True,exist_ok=True)
    inputs={}
    def read(role,relative):
        path=roots[role]/relative
        inputs[f'{role}/{relative}']=sha(path)
        return pd.read_csv(path)

    pred=read('release_root','expected/posthoc/results/outer_predictions.csv')
    expected=read('release_root','expected/posthoc/results/repeat_results.csv')
    by_model=[];by_pair=[];pool_model=[];pool_pair=[];pediatric_errors=[]
    for design,base,aug in PAIRS:
        z=paired_frame(pred,base,aug,['repeat','outer_fold','participant_id'])
        assert len(z)==960 and z.participant_id.nunique()==96
        assert not z.duplicated(['repeat','participant_id']).any()
        assert set(z.repeat)==set(range(10))
        pairrows=[];modelrows=[]
        for repeat,d in z.groupby('repeat',sort=True):
            assert len(d)==96 and int(d.y_base.sum())==41
            context={'design':design,'baseline':base,'augmented':aug,'repeat':int(repeat)}
            row=context|paired_summary(d.y_base,d.p_base,d.p_add)
            by_pair.append(row);pairrows.append(row)
            for role,column,model in [('baseline','p_base',base),('augmented','p_add',aug)]:
                rr=context|{'role':role,'model':model}|probability_summary(d.y_base,d[column])
                by_model.append(rr);modelrows.append(rr)
            target=expected[(expected.baseline==base)&(expected.augmented==aug)&(expected.repeat==repeat)]
            assert len(target)==1
            for metric in ['baseline_logloss','augmented_logloss','delta_ll','delta_brier','delta_auroc']:
                error=abs(row[metric]-float(target.iloc[0][metric]))
                assert error<TOL,(design,repeat,metric,error)
                pediatric_errors.append(error)
        # Pooled quantiles/counts describe 960 dependent participant-repeat rows.
        # AUROC is always the mean of the ten repeat-level AUROCs, never pooled.
        context={'design':design,'baseline':base,'augmented':aug,'repeats':10,'unique_participants':96}
        rr=context|paired_summary(z.y_base,z.p_base,z.p_add)
        rr['delta_auroc']=float(np.mean([r['delta_auroc'] for r in pairrows]))
        pool_pair.append(rr)
        for role,column,model in [('baseline','p_base',base),('augmented','p_add',aug)]:
            rr=context|{'role':role,'model':model}|probability_summary(z.y_base,z[column])
            rr['auroc']=float(np.mean([r['auroc'] for r in modelrows if r['role']==role]))
            rr['tail_unique_participants']=int(z.loc[wrong_tail(z.y_base,z[column]),'participant_id'].nunique())
            pool_model.append(rr)
    save(out/'PEDIATRIC_PROBABILITY_BY_REPEAT.csv',by_model)
    save(out/'PEDIATRIC_PAIR_BY_REPEAT.csv',by_pair)
    save(out/'PEDIATRIC_PROBABILITY_SUMMARY.csv',pool_model)
    save(out/'PEDIATRIC_PAIR_SUMMARY.csv',pool_pair)

    tp=read('transfer_root','analysis/results/PREDICTIONS.csv')
    td=read('transfer_root','analysis/results/CONDITIONAL_DRAWS.csv')
    trng=np.random.default_rng(20261103)
    trows=[];tmodel=[];tcheck=[];indices={'transfer':{},'adult':{}}
    for direction,frame in tp.groupby('direction',sort=False):
        base=frame[frame.model=='baseline'].set_index('participant_id').sort_index()
        y=base.y.to_numpy(int);p0=base.p.to_numpy(float)
        # This order and vectorized shape reproduce the preserved generator stream.
        groups=[np.flatnonzero(y==c) for c in [0,1]]
        ix=np.column_stack([trng.choice(g,(5000,len(g)),replace=True) for g in groups])
        indices['transfer'][direction]={str(i):ix[i].tolist() for i in TRANSFER_DRAWS}
        for model in FOCAL:
            add=frame[frame.model==model].set_index('participant_id').loc[base.index]
            assert np.array_equal(y,add.y.to_numpy(int))
            p1=add.p.to_numpy(float)
            source_device=str(base.source_device.iloc[0])
            source=tp[(tp.model=='baseline')&(tp.target_device==source_device)]
            ctx={'direction':direction,'model':model,'source_device':source_device,
                 'target_device':str(base.target_device.iloc[0]),'source_n':len(source),
                 'source_prevalence':float(source.y.mean()),'target_n':len(y)}
            trows.append(ctx|paired_summary(y,p0,p1))
            for role,p in [('baseline',p0),('augmented',p1)]:
                tmodel.append(ctx|{'role':role}|probability_summary(y,p))
            saved=td[(td.direction==direction)&(td.model==model)].set_index('draw')
            for i in TRANSFER_DRAWS:
                j=ix[i];v=paired_summary(y[j],p0[j],p1[j]);r=ctx|{'draw':i}
                for k in ['delta_ll','delta_brier','delta_auroc','equal_class_delta_ll']:
                    r[k+'_replayed']=v[k];r[k+'_saved']=float(saved.loc[i,k])
                    r[k+'_abs_error']=abs(v[k]-r[k+'_saved'])
                    assert r[k+'_abs_error']<TOL,(direction,model,i,k)
                tcheck.append(r)
    save(out/'TRANSFER_PAIR_SUMMARY.csv',trows)
    save(out/'TRANSFER_PROBABILITY_SUMMARY.csv',tmodel)
    save(out/'TRANSFER_SELECTED_BOOTSTRAP_REPLAY.csv',tcheck)

    ap=read('extension_root','results/external/PREDICTIONS.csv')
    ad=read('extension_root','results/external/CONDITIONAL_RESAMPLES.csv')
    yframe=ap[['source_test_row','y']].drop_duplicates().sort_values('source_test_row')
    assert len(yframe)==36 and yframe.source_test_row.nunique()==36
    y=yframe.y.to_numpy(int);assert np.array_equal(np.bincount(y),[18,18])
    arng=np.random.default_rng(20261137)
    ix=np.column_stack([arng.choice(np.flatnonzero(y==c),size=(10000,int(sum(y==c))),replace=True) for c in [0,1]])
    indices['adult']={str(i):ix[i].tolist() for i in ADULT_DRAWS}
    acheck=[]
    for (baseline,learner),d in ap.groupby(['baseline','learner'],sort=True):
        z=paired_frame(d,'baseline','eeg_augmented',['source_test_row'])
        assert np.array_equal(z.y_base.to_numpy(int),y)
        p0=z.p_base.to_numpy(float);p1=z.p_add.to_numpy(float)
        saved=ad[(ad.baseline==baseline)&(ad.learner==learner)].set_index('bootstrap')
        for i in ADULT_DRAWS:
            j=ix[i];v=paired_summary(y[j],p0[j],p1[j]);r={'baseline':baseline,'learner':learner,'draw':i}
            for original,computed in [('delta_logloss','delta_ll'),('delta_brier','delta_brier'),('delta_auroc','delta_auroc')]:
                r[original+'_replayed']=v[computed];r[original+'_saved']=float(saved.loc[i,original])
                r[original+'_abs_error']=abs(v[computed]-r[original+'_saved'])
                assert r[original+'_abs_error']<TOL,(baseline,learner,i,original)
            acheck.append(r)
    assert len(tcheck)==20 and len(acheck)==30
    save(out/'ADULT_SELECTED_BOOTSTRAP_REPLAY.csv',acheck)
    dump(out/'SELECTED_SAMPLING_INDICES.json',indices)
    for name,before in inputs.items():
        role,relative=name.split('/',1)
        assert sha(roots[role]/relative)==before,'Input changed during execution'
    maxerr=lambda rows:max(float(v) for r in rows for k,v in r.items() if k.endswith('_abs_error'))
    verification=dict(status='passed',analysis_status='post hoc; non-blind; fixed saved predictions',
                      predictive_model_fits=0,calibration_model_fits=0,raw_measurement_extractions=0,
                      pediatric_pairs=3,pediatric_pair_repeat_rows=30,pediatric_model_repeat_rows=60,
                      pediatric_max_metric_abs_error=max(pediatric_errors),
                      transfer_selected_draws=TRANSFER_DRAWS,transfer_comparison_draw_rows=len(tcheck),
                      transfer_max_abs_error=maxerr(tcheck),adult_selected_draws=ADULT_DRAWS,
                      adult_comparison_draw_rows=len(acheck),adult_max_abs_error=maxerr(acheck),
                      full_bootstrap_metric_replay=False,fixed_oof_endpoint_replay=False,
                      tail_thresholds={'positive_class_incorrect_p_lt':.1,'negative_class_incorrect_p_gt':.9},
                      note='Index matrices reconstructed for exact random streams; metrics compared only for selected draws. Saved draw files store metrics, not sampling indices.',
                      input_hashes_unchanged=True)
    dump(out/'VERIFICATION.json',verification)
    dump(out/'INPUT_MANIFEST.json',dict(inputs=inputs,code_sha256=sha(__file__),
                                     analysis_plan_sha256=sha(Path(__file__).with_name('ANALYSIS_PLAN.txt')),
                                     software={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__}))
    write_text_outputs(out,pool_model,pool_pair,trows,verification)
    print(json.dumps(verification,indent=2))

def write_text_outputs(out,models,pairs,transfer,verification):
    labels={'unadjusted':'Unadjusted','device':'Device','device_quality':'Device + quality'}
    parts=[r'\section{Post-hoc probability-location and tail diagnostics}',
           r'These descriptive analyses use the existing held-out probabilities without refitting or recalibration. The pediatric comparisons retain all 96 participants and ten dependent repeats. Scores are calculated within repeats before averaging; probability summaries and tail counts summarize the 960 dependent participant--repeat rows. The observed ADHD study-group fraction is $41/96=0.4271$. A high-confidence incorrect probability is defined here as $p<0.1$ for an observed positive label or $p>0.9$ for an observed negative label. This fixed rule describes probability tails and is not a clinical decision threshold.',
           r'\begin{table}[htbp]\centering\small',
           r'\caption{Saved-probability diagnostics for entropy/Hjorth augmentation. B and E denote the matched baseline and augmented models. The final column is the union-tail contribution to the overall paired log-loss increment.}\label{tab:probability-tail-diagnostics}',
           r'\begin{tabular}{lrrrrrrr}\toprule',
           r'Design & Mean $p_B$ & Mean $p_E$ & $\Delta$LL & $\Delta$Brier & $\Delta$AUC & Tails B/E & Tail $\Delta$LL\\\midrule']
    for r in pairs:
        m=[x for x in models if x['design']==r['design']]
        n0=next(x['tail_wrong_total'] for x in m if x['role']=='baseline')
        n1=next(x['tail_wrong_total'] for x in m if x['role']=='augmented')
        parts.append(f"{labels[r['design']]} & {r['baseline_mean_p']:.4f} & {r['augmented_mean_p']:.4f} & {r['delta_ll']:+.4f} & {r['delta_brier']:+.4f} & {r['delta_auroc']:+.4f} & {n0}/{n1} & {r['tail_union_delta_ll_contribution']:+.4f}"+r'\\')
    parts.extend([r'\bottomrule\end{tabular}',
                  r'\par\smallskip\begin{minipage}{\textwidth}\footnotesize Tail counts are occurrences among 960 participant--repeat predictions per model, not independent participants. Union-tail contributions sum paired loss changes for rows meeting the rule in either model, divided by all 960 rows; the complementary contribution is retained in the accompanying CSV. Positive increments favor EEG. Device-plus-quality adjustment includes EEG-derived measures in both models.\end{minipage}',r'\end{table}'])
    dq=next(r for r in pairs if r['design']=='device_quality')
    parts.append(f"After device-plus-quality adjustment, the mean log-loss increment is ${dq['delta_ll']:+.4f}$; the union-tail and complementary-row contributions are ${dq['tail_union_delta_ll_contribution']:+.4f}$ and ${dq['tail_neither_delta_ll_contribution']:+.4f}$, respectively. These are an algebraic description of the fixed paired losses, not a causal decomposition or evidence for an independently validated subgroup. Mean probabilities and tail summaries do not establish calibrated predictions.")
    parts.extend([r'\begin{table}[htbp]\centering\small',
                  r'\caption{Probability location in the device-transfer comparisons. Predictions remain unchanged.}\label{tab:transfer-probability-location}',
                  r'\begin{tabular}{llrrrrr}\toprule',
                  r'Train $\to$ assess & Procedure & Target fraction & Mean $p_B$ & Mean $p_E$ & $\Delta$LL & Equal-class $\Delta$LL\\\midrule'])
    for r in transfer:
        direction=r'X $\to$ Plus' if r['direction']=='X_to_Plus' else r'Plus $\to$ X'
        model='Selector' if r['model']=='spectral_covariance_selector' else 'Entropy/Hjorth'
        parts.append(f"{direction} & {model} & {r['observed_prevalence']:.4f} & {r['baseline_mean_p']:.4f} & {r['augmented_mean_p']:.4f} & {r['delta_ll']:+.4f} & {r['equal_class_delta_ll']:+.4f}"+r'\\')
    parts.extend([r'\bottomrule\end{tabular}',r'\end{table}',
                  r'In both transfer directions, EEG augmentation moves the mean predicted probability toward the target group fraction. This does not identify an intercept-only explanation or a device-invariant mechanism. Equal-class weighting changes the assessment weighting and does not recalibrate predictions or remove population shift.',
                  r'A bounded independent implementation also reproduced five selected saved bootstrap draws for each of the four focal transfer comparisons and six adult comparisons, using the original seeds and shared class-stratified sampling scheme. It reconstructed sampling indices but evaluated metrics only at the selected indices; it did not rerun complete bootstrap distributions or any predictive model.'])
    (out/'SUGGESTED_SECTION.tex').write_text('\n\n'.join(parts)+'\n',encoding='utf-8')
    finding=['Post-hoc saved-prediction findings','',
             'All rows and thresholds were retained as planned. No predictive or calibration model was fitted. Counts below are occurrences among 960 dependent participant-repeat predictions per pediatric model.','']
    for r in pairs:
        m=[x for x in models if x['design']==r['design']]
        tails='/'.join(str(next(x['tail_wrong_total'] for x in m if x['role']==role)) for role in ['baseline','augmented'])
        finding.append(f"{labels[r['design']]}: mean probabilities B/E {r['baseline_mean_p']:.6f}/{r['augmented_mean_p']:.6f}; delta LL {r['delta_ll']:+.6f}, Brier {r['delta_brier']:+.6f}, AUROC {r['delta_auroc']:+.6f}; incorrect tails B/E {tails}; union-tail contribution {r['tail_union_delta_ll_contribution']:+.6f}, remaining-row contribution {r['tail_neither_delta_ll_contribution']:+.6f}.")
    finding.extend(['','Probability summaries are descriptive; closeness of mean probability to prevalence is not sufficient for calibration. The tail partition is an exact accounting of fixed loss changes, not a biological or causal explanation.','',
                    f"Verification: 30 pediatric pair-repeat rows agree with saved metrics (maximum error {verification['pediatric_max_metric_abs_error']:.3g}); 20 transfer comparison/draw rows agree (maximum error {verification['transfer_max_abs_error']:.3g}); 30 adult comparison/draw rows agree (maximum error {verification['adult_max_abs_error']:.3g}).",
                    'Full bootstrap distributions and original fixed-OOF endpoints were not replayed in this follow-up. No raw signals, new labels, or independently collected participants were analyzed.'])
    (out/'FINDINGS.txt').write_text('\n'.join(finding)+'\n',encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-root',required=True)
    parser.add_argument('--transfer-root',required=True)
    parser.add_argument('--extension-root',required=True)
    parser.add_argument('--output',required=True)
    run(parser.parse_args())
