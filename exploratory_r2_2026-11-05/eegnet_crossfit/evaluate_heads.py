from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
from sklearn.model_selection import StratifiedKFold

HERE=Path(__file__).resolve().parent

def tuned_head(x,y,z,seed):
    k=min(5,int(np.bincount(y,minlength=2).min()))
    assert k>=2
    split=list(StratifiedKFold(k,shuffle=True,random_state=seed).split(x,y))
    cache=[(*scale(x[it],x[iv]),it,iv) for it,iv in split]
    scores={}
    for c in GRID:
        pooled=[]
        for a,b,it,iv in cache:
            pooled.extend(loss(y[iv],fit_predict(a,y[it],b,c)))
        scores[c]=float(np.mean(pooled))
    best=min(GRID,key=lambda c:(round(scores[c],12),c))
    a,b=scale(x,z)
    return fit_predict(a,y,b,best),best,scores,k

def main():
    assert (HERE/'INFERENCE_COMPLETE.json').exists()
    base,linked,e4,nuisance=baseline_data();ids=base.index.tolist()
    y=linked.group_y.to_numpy(int);x=base[B].to_numpy(float)
    lookup={pid:i for i,pid in enumerate(ids)}
    manifest=read(REPO/'audit/preregistered_analysis/PRIMARY_FOLD_MANIFEST.csv')
    old=read(REPO/'companion/extensions/eegnet/results/OUTER_PREDICTIONS.csv')
    out=HERE/'outputs';out.mkdir(exist_ok=True)
    predictions=[];tuning=[];roster_checks=[]
    for r,f,tr,te,folds in fold_records(manifest,ids):
        te_ids=[ids[i] for i in te];oof=np.full(len(ids),np.nan)
        components=[];test_logits=[]
        for j,(it,iv) in enumerate(folds):
            source=HERE/'scores'/f'r{r:02d}_f{f:02d}_i{j}.npz'
            with np.load(source,allow_pickle=False) as z:
                vi=z['head_train_ids'].tolist();eti=z['encoder_train_ids'].tolist()
                tsi=z['test_ids'].tolist()
                logits=z['head_train_logits'];tested=z['test_logits']
            assert set(vi)=={ids[i] for i in iv}
            assert set(eti)=={ids[i] for i in it}
            assert not set(vi)&set(eti) and not set(tsi)&(set(vi)|set(eti))
            assert set(tsi)==set(te_ids)
            head_idx=np.array([lookup[p] for p in vi])
            tested=tested[[tsi.index(p) for p in te_ids]]
            oof[head_idx]=logits;test_logits.append(tested)
            pair={}
            for name,xt,zt in [('baseline',x[head_idx],x[te]),
                               ('eegnet',np.column_stack([x[head_idx],logits]),np.column_stack([x[te],tested]))]:
                pair[name],best,scores,k=tuned_head(xt,y[head_idx],zt,20261105+r*1000+f*10+j)
                for c in GRID:
                    tuning.append(dict(repeat=r,outer_fold=f,encoder_inner_fold=j,model=name,
                        C=c,inner_logloss=scores[c],selected=c==best,head_train_n=len(head_idx),cv_splits=k))
            components.append(pair)
            roster_checks.append(dict(repeat=r,outer_fold=f,inner_fold=j,
                head_train_n=len(head_idx),encoder_train_n=len(it),test_n=len(te),
                head_participant_encoder_overlap=0,test_participant_training_overlap=0))
        assert np.isfinite(oof[tr]).all() and np.isnan(oof[te]).all()
        pooled_raw=np.column_stack([x,oof])
        a,b=scale(x[tr],x[te]);pb=fit_predict(a,y[tr],b,1.)
        raw_train=pooled_raw[tr]
        sc=StandardScaler().fit(raw_train)
        # One fixed-C head; concatenate five assessment components into one call.
        assessments=np.concatenate([np.column_stack([x[te],z]) for z in test_logits],axis=0)
        pe=fit_predict(sc.transform(raw_train),y[tr],sc.transform(assessments),1.).reshape(5,len(te)).mean(axis=0)
        eb=np.mean([p['baseline'] for p in components],axis=0)
        ee=np.mean([p['eegnet'] for p in components],axis=0)
        recorded=old[old.repeat.eq(r)&old.outer_fold.eq(f)].set_index('participant_id').loc[te_ids]
        for j,idx in enumerate(te):
            predictions.append(dict(repeat=r,outer_fold=f,participant_id=ids[idx],y=int(y[idx]),
                p_ensemble_baseline=float(eb[j]),p_ensemble_eegnet=float(ee[j]),
                p_pooled_baseline=float(pb[j]),p_pooled_eegnet=float(pe[j]),
                p_recorded_baseline=float(recorded.p_baseline.iloc[j]),
                p_recorded_eegnet=float(recorded.p_eegnet.iloc[j])))
        pd.DataFrame(predictions).to_csv(out/'OUTER_PREDICTIONS.csv',index=False,float_format='%.17g')
        if f==4:print(f'Cross-fitted heads: repeat {r+1}/10',flush=True)
    pred=pd.DataFrame(predictions)
    specs=[('p_ensemble_baseline','p_ensemble_eegnet','split_ensemble'),
           ('p_recorded_baseline','p_ensemble_eegnet','split_ensemble_vs_recorded_full_baseline'),
           ('p_pooled_baseline','p_pooled_eegnet','pooled_oof_fixed_C1'),
           ('p_recorded_baseline','p_pooled_eegnet','pooled_oof_vs_recorded_full_baseline'),
           ('p_recorded_baseline','p_recorded_eegnet','original_embedding_reference')]
    summaries=[summarize(pred,b,a,n,out) for b,a,n in specs]
    pd.DataFrame(tuning).to_csv(out/'HEAD_TUNING.csv',index=False,float_format='%.17g')
    pd.DataFrame(roster_checks).to_csv(out/'EXCLUSION_CHECKS.csv',index=False)
    pd.DataFrame(summaries).to_csv(out/'SUMMARY.csv',index=False,float_format='%.17g')
    provenance(out,[HERE/'PLAN_2026-11-05.md',Path(__file__),HERE/'INFERENCE_COMPLETE.json',
        REPO/'audit/preregistered_analysis/PRIMARY_FOLD_MANIFEST.csv',
        REPO/'audit/stage1_blind_feature_extraction/BLIND_PRIMARY_FEATURE_MATRIX.csv'],
        {'new_encoder_fits':0,'encoders_reused':250,'head_self_encoder_overlap':0,
         'split_head_training_min':min(z['head_train_n'] for z in roster_checks),
         'split_head_training_max':max(z['head_train_n'] for z in roster_checks)})
    print(pd.DataFrame(summaries).to_string(index=False))

if __name__=='__main__':main()
