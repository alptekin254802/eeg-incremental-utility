"""Verify and display the complete saved EEGNet comparison; no training."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare_windows import write,sha

def loss(y,p):
    p=np.clip(p,1e-8,1-1e-8)
    return -y*np.log(p)-(1-y)*np.log1p(-p)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--release',type=Path,required=True);a=ap.parse_args()
    if a.out.exists() and any(a.out.iterdir()):raise FileExistsError('Output must be empty')
    a.out.mkdir(parents=True,exist_ok=True)
    p=pd.read_csv(a.results/'OUTER_PREDICTIONS.csv');c=pd.read_csv(a.results/'INNER_CANDIDATES.csv')
    assert len(p)==960 and len(c)==500 and not p.duplicated(['repeat','participant_id']).any()
    assert p.groupby('repeat').size().eq(96).all() and p.groupby('participant_id').size().eq(10).all()
    labels=p.groupby('participant_id').y.first().sort_index();assert labels.sum()==41 and len(labels)==96
    base=pd.read_csv(a.release/'audit/stage1_blind_feature_extraction/BLIND_PRIMARY_FEATURE_MATRIX.csv').set_index('participant_id').iloc[:,:7]
    frozen=pd.read_csv(a.release/'audit/preregistered_analysis/PRIMARY_OUTER_PREDICTIONS.csv');frozen=frozen[frozen.outcome.eq('group')].set_index(['repeat','participant_id'])
    curves=[];error=0.;scores_error=0.;baseline_error=0.;encoders=0
    for r in range(10):
        for f in range(5):
            folder=a.results/f'r{r:02d}_f{f:02d}';pp=p[p.repeat.eq(r)&p.outer_fold.eq(f)].set_index('participant_id')
            final=json.loads((folder/'outer/fit.json').read_text());test=set(final['assessment_ids']);tr=set(final['train_ids'])
            assert not test&tr and len(test|tr)==96 and set(pp.index)==test
            scores={kind:{value:[] for value in (.01,.1,1.,10.,100.)} for kind in ('baseline','eegnet')}
            validated=[]
            for name in [f'inner{i}' for i in range(5)]+['outer']:
                meta=json.loads((folder/name/'fit.json').read_text());encoders+=1
                assert meta['epochs']==50 and len(meta['loss'])==50 and np.isfinite(meta['loss']).all()
                assert meta['parameters']==1457
                curves.append(dict(repeat=r,outer_fold=f,partition=name,loss=meta['loss']))
                it,iv=set(meta['train_ids']),set(meta['assessment_ids']);assert not it&iv
                if name!='outer':
                    assert it|iv==tr and not (it|iv)&test;validated+=meta['assessment_ids']
                    with np.load(folder/name/'embeddings.npz',allow_pickle=False) as iz:
                        assert iz['participant_id'].tolist()==meta['train_ids']+meta['assessment_ids']
                        ez=pd.DataFrame(iz['embedding'],index=iz['participant_id'])
                    for kind in scores:
                        for value in scores[kind]:
                            v=json.loads((folder/name/f'{kind}_C{value:g}.json').read_text())
                            assert set(v['validation_ids'])==iv
                            xx=base.loc[v['validation_ids']].to_numpy()
                            if kind=='eegnet':xx=np.column_stack([xx,ez.loc[v['validation_ids']].to_numpy()])
                            h=v['head'];replayed=expit(((xx-np.asarray(h['mean']))/np.asarray(h['scale']))@np.asarray(h['coef'])[0]+h['intercept'][0])
                            error=max(error,float(np.max(np.abs(replayed-np.asarray(v['probabilities'])))))
                            yy=labels.loc[v['validation_ids']].to_numpy();scores[kind][value].extend(loss(yy,np.asarray(v['probabilities'])))
            assert len(validated)==len(tr) and set(validated)==tr
            for kind in scores:
                means={value:float(np.mean(v)) for value,v in scores[kind].items()}
                expected=min(means,key=lambda v:(round(means[v],12),v))
                cc=c[c.repeat.eq(r)&c.outer_fold.eq(f)&c.model.eq(kind)]
                assert len(cc)==5 and float(cc[cc.selected].C.iloc[0])==expected
                for row in cc.itertuples():scores_error=max(scores_error,abs(row.inner_logloss-means[row.C]))
            with np.load(folder/'outer/embeddings.npz',allow_pickle=False) as z:
                ids=z['participant_id'].tolist();emb=z['embedding'];assert ids==final['train_ids']+final['assessment_ids']
                assert emb.shape==(96,128) and np.isfinite(emb).all()
            # Numerical head replay uses stored transformed model parameters and frozen baseline predictors.
            head=json.loads((folder/'outer/eegnet_head.json').read_text())
            assert len(head['mean'])==135 and np.isfinite(head['coef']).all()
            for kind in ('baseline','eegnet'):
                h=json.loads((folder/'outer'/f'{kind}_head.json').read_text());xx=base.loc[final['assessment_ids']].to_numpy()
                if kind=='eegnet':xx=np.column_stack([xx,emb[len(final['train_ids']):]])
                replayed=expit(((xx-np.asarray(h['mean']))/np.asarray(h['scale']))@np.asarray(h['coef'])[0]+h['intercept'][0])
                error=max(error,float(np.max(np.abs(replayed-pp.loc[final['assessment_ids'],'p_'+kind].to_numpy()))))
            for pid,row in pp.iterrows():baseline_error=max(baseline_error,abs(row.p_baseline-frozen.loc[(r,pid),'p_m2']))
            baseline_error=max(baseline_error,json.loads((folder/'COMPLETE.json').read_text())['baseline_max_abs_error'])
    assert encoders==300 and scores_error<1e-12 and baseline_error<1e-7 and error<1e-12
    rows=[]
    for r,g in p.groupby('repeat'):
        y=g.y.to_numpy();record={'repeat':int(r)}
        for name in ('baseline','eegnet'):
            prob=g['p_'+name].to_numpy()
            record[name+'_logloss']=float(loss(y,prob).mean());record[name+'_auroc']=float(roc_auc_score(y,prob));record[name+'_brier']=float(np.mean((y-prob)**2))
        for metric in ('logloss','brier'):record['delta_'+metric]=record['baseline_'+metric]-record['eegnet_'+metric]
        record['delta_auroc']=record['eegnet_auroc']-record['baseline_auroc'];rows.append(record)
    repeats=pd.DataFrame(rows);repeats.to_csv(a.out/'METRICS_BY_REPEAT.csv',index=False,float_format='%.17g')
    d=repeats.delta_logloss.to_numpy()
    summary={key:float(repeats[key].mean()) for key in repeats if key!='repeat'}
    summary['repeat_logloss']=dict(median=float(np.median(d)),q25=float(np.quantile(d,.25)),q75=float(np.quantile(d,.75)),min=float(d.min()),max=float(d.max()),positive=int(sum(d>0)),n=10)
    p['delta_ll']=loss(p.y.to_numpy(),p.p_baseline.to_numpy())-loss(p.y.to_numpy(),p.p_eegnet.to_numpy())
    p['delta_brier']=(p.y-p.p_baseline)**2-(p.y-p.p_eegnet)**2
    means=p.groupby('participant_id')[['delta_ll','delta_brier']].mean().loc[labels.index]
    rng=np.random.default_rng(20261102);group=[np.flatnonzero(labels.to_numpy()==i) for i in (0,1)]
    draws=[]
    for b in range(10000):
        sampled=np.concatenate([rng.choice(ix,len(ix),replace=True) for ix in group])
        draws.append(means.to_numpy()[sampled].mean(0))
    draws=np.asarray(draws);pd.DataFrame(draws,columns=['delta_logloss','delta_brier']).to_csv(a.out/'CONDITIONAL_BOOTSTRAP.csv',index=False,float_format='%.17g')
    summary['conditional_intervals']={m:np.quantile(draws[:,j],[.025,.975]).tolist() for j,m in enumerate(('delta_logloss','delta_brier'))}
    summary['interval_scope']='Stratified participant resampling conditional on fixed fitted predictions; excludes training and selection uncertainty'
    summary['training_loss']={part:{'median_first':float(np.median([x['loss'][0] for x in curves if (x['partition']=='outer')==(part=='outer')])), 'median_last':float(np.median([x['loss'][-1] for x in curves if (x['partition']=='outer')==(part=='outer')]))} for part in ('inner','outer')}
    write(a.out/'SUMMARY.json',summary)
    write(a.out/'VERIFICATION.json',dict(status='PASS',encoders=encoders,epochs_per_encoder=50,paired_predictions=960,inner_candidates=500,maximum_inner_score_error=scores_error,maximum_head_probability_error=error,maximum_frozen_baseline_prediction_error=baseline_error,fold_membership_checked=True,scope='Saved-result arithmetic, all 2600 logistic-head predictions, and membership verification; encoder replay is separately recorded'))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,ax=plt.subplots(1,2,figsize=(7.3,3.2))
    ax[0].axhline(0,color='.55',lw=.8);ax[0].plot(np.arange(1,11),d,'o-',color='#176b80',lw=1)
    ax[0].axhline(d.mean(),color='#bd5c30',ls='--',lw=1,label='Mean')
    ax[0].set(xlabel='Cross-validation repeat',ylabel='Log-loss increment',xticks=np.arange(1,11),title='A  Held-out increment');ax[0].legend(frameon=False)
    for x in curves:ax[1].plot(np.arange(1,51),x['loss'],color='#176b80' if x['partition']=='outer' else '.55',alpha=.07,lw=.6)
    for part,color,label in [('inner','.35','Inner fits (250)'),('outer','#176b80','Outer fits (50)')]:
        series=np.asarray([x['loss'] for x in curves if (x['partition']=='outer')==(part=='outer')]);ax[1].plot(np.arange(1,51),np.median(series,axis=0),color=color,lw=1.6,label=label)
    ax[1].set(xlabel='Epoch',ylabel='Sampled-bag cross entropy',title='B  Fixed-budget training');ax[1].legend(frameon=False)
    for item in ax:item.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(a.out/'eegnet_comparison.pdf',metadata={'CreationDate':None,'ModDate':None});fig.savefig(a.out/'eegnet_comparison.svg');plt.close(fig)
    lines=[r'\begin{table}[htbp]',r'\centering',r'\caption{Exploratory EEGNet-like representation comparison. Means average ten dependent repeats. Positive increments favor augmentation. Conditional percentile ranges resample participants with the fitted predictions held fixed; they omit training and selection uncertainty.}',r'\label{tab:eegnet}',r'\begin{tabular}{lrrrr}',r'\toprule',r'Measure & Baseline & EEGNet-like & Increment & Conditional range\\',r'\midrule']
    for metric,label in [('logloss','Log loss'),('auroc','AUROC'),('brier','Brier score')]:
        ci=summary['conditional_intervals'].get('delta_'+metric)
        interval=f'$[{ci[0]:+.4f}, {ci[1]:+.4f}]$' if ci else '---'
        lines.append(f"{label} & ${summary['baseline_'+metric]:.4f}$ & ${summary['eegnet_'+metric]:.4f}$ & ${summary['delta_'+metric]:+.4f}$ & {interval}"+r'\\')
    lines.extend([r'\bottomrule\end{tabular}',r'\end{table}'])
    (a.out/'eegnet_summary.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(a.out/'RESULT_IDENTITIES.json',{f.relative_to(a.results).as_posix():sha(f) for f in sorted(a.results.rglob('*')) if f.is_file()})
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
