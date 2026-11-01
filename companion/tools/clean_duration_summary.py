"""Summarize recorded retained EEG duration; no signal processing or model fitting."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); root=a.release_root.resolve(); out=a.output.resolve()
    if out.is_relative_to(root):p.error('Output must be outside the source release.')
    if out.exists() and any(out.iterdir()):p.error('Output must be new or empty.')
    paths=['audit/stage1_blind_feature_extraction/EEG_PREPROCESSING_QC.csv',
           'audit/preregistered_analysis/OUTCOME_LINKAGE_MANIFEST.csv']
    q,l=[pd.read_csv(root/x,float_precision='round_trip') for x in paths]
    d=l.merge(q,on='participant_id',validate='one_to_one')
    assert len(d)==96 and set(d.participant_id)==set(q.loc[q.preprocessing_status=='PASS','participant_id'])
    assert (d.clean_seconds>=120).all() and (d.clean_window_fraction>=.5).all()
    assert np.allclose(d.clean_window_fraction,(d.candidate_window_count-d.rejected_window_count)/d.candidate_window_count,atol=1e-15,rtol=0)
    d['retained_minutes']=d.clean_seconds/60
    d['group']=d.group_y.map({0:'Control',1:'ADHD study group'})
    rows=[]
    groups=[('All participants',d)]
    groups += [(name,g) for name,g in d.groupby('group',sort=True)]
    groups += [(name,g) for name,g in d.groupby('device',sort=True)]
    groups += [(f'{device} / {group}',g) for (device,group),g in d.groupby(['device','group'],sort=True)]
    for label,g in groups:
        row={'stratum':label,'n':len(g)}
        for col in ['retained_minutes','clean_window_fraction']:
            v=g[col].to_numpy()
            for name,val in zip(['min','q1','median','q3','max'],[v.min(),*np.quantile(v,[.25,.5,.75],method='linear'),v.max()]):row[f'{col}_{name}']=float(val)
        rows.append(row)
    out.mkdir(parents=True,exist_ok=True)
    summary=pd.DataFrame(rows)
    summary.to_csv(out/'CLEAN_DURATION_SUMMARY.csv',index=False,float_format='%.17g')
    d[['participant_id','group_y','device','clean_seconds','retained_minutes','clean_window_fraction','candidate_window_count','rejected_window_count']].sort_values('participant_id').to_csv(out/'CLEAN_DURATION_PARTICIPANTS.csv',index=False,float_format='%.17g')
    text=r'''\begin{table}[!htbp]
\centering\small
\caption{Recorded retained EEG duration and clean-window fraction in the final cohort.}
\label{tab:duration}
\begin{tabular}{lrrrr}
\toprule
Stratum & $n$ & Minutes, median [IQR] & Range & Clean fraction, median [IQR]\\
\midrule
'''
    for r in rows:
        k='retained_minutes_';f='clean_window_fraction_'
        label=r['stratum'].replace('ADHD study group','ADHD group')
        text+=f"{label} & {r['n']} & {r[k+'median']:.2f} [{r[k+'q1']:.2f}, {r[k+'q3']:.2f}] & {r[k+'min']:.2f}--{r[k+'max']:.2f} & {r[f+'median']:.3f} [{r[f+'q1']:.3f}, {r[f+'q3']:.3f}]"+r'\\'+'\n'
    text+=r'''\bottomrule
\end{tabular}
\par\smallskip\begin{minipage}{\textwidth}\footnotesize
Duration is the recorded union of accepted window intervals, with overlapping samples counted once. IQR gives the 25th and 75th percentiles (linear interpolation). Clean fraction is accepted divided by candidate windows. The strata overlap; they are descriptive summaries, not independent comparisons or evidence that all residual artifacts were removed.
\end{minipage}
\end{table}
'''
    (out/'clean_duration.tex').write_text(text,encoding='utf-8')
    (out/'VERIFICATION.json').write_text(json.dumps({'scope':'Saved-record descriptive recomputation; no raw EEG processing or model fitting','participants':len(d),'source_sha256':{x:hashlib.sha256((root/x).read_bytes()).hexdigest() for x in paths},'quartiles':'numpy quantile method=linear, type 7','cohort_and_fraction_checks':True},indent=2),encoding='utf-8')
    print(summary.to_string(index=False))

if __name__=='__main__':main()
