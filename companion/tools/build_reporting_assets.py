"""Build the supplementary reporting tables from preserved derived records."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd

def build(repository,output):
    root=Path(repository).resolve();out=Path(output).resolve()
    if out.is_relative_to(root):raise ValueError('Output must be outside the repository')
    paths={
        'quality':'article_assets/quality_summary/CLEAN_DURATION_SUMMARY.csv',
        'retention':'reporting/provenance/RETENTION_PROFILE.csv',
        'pairs':'reporting/probability_diagnostics/results/PEDIATRIC_PAIR_SUMMARY.csv',
        'transfer':'reporting/probability_diagnostics/results/TRANSFER_PAIR_SUMMARY.csv'}
    data={k:pd.read_csv(root/v,float_precision='round_trip') for k,v in paths.items()}
    tables=out/'tables';tables.mkdir(parents=True,exist_ok=True)
    def table(name,caption,label,cols,head,rows,note):
        text='\\begin{table}[!htbp]\n\\centering\\small\n\\caption{'+caption+'}\n\\label{'+label+'}\n'
        text+='\\begin{tabular}{'+cols+'}\n\\toprule\n'+head+'\\\\\n\\midrule\n'
        text+='\n'.join(' & '.join(row)+r'\\' for row in rows)
        text+='\n\\bottomrule\n\\end{tabular}\n'
        text+='\\par\\smallskip\\begin{minipage}{\\textwidth}\\footnotesize '+note+'\n\\end{minipage}\n\\end{table}\n'
        (tables/(name+'.tex')).write_text(text,encoding='utf-8')
    rows=[]
    for _,r in data['quality'].iterrows():
        k='retained_minutes_';f='clean_window_fraction_'
        rows.append([r.stratum.replace('ADHD study group','ADHD group'),str(int(r.n)),f"{r[k+'median']:.2f} [{r[k+'q1']:.2f}, {r[k+'q3']:.2f}]",f"{r[k+'min']:.2f}--{r[k+'max']:.2f}",f"{r[f+'median']:.3f} [{r[f+'q1']:.3f}, {r[f+'q3']:.3f}]"])
    table('clean_duration','Recorded retained EEG duration and clean-window fraction in the final cohort.','tab:duration','lrrrr',r'Stratum & $n$ & \shortstack{Minutes\\Median [IQR]} & Range & \shortstack{Clean fraction\\Median [IQR]}',rows,'Duration is the recorded union of accepted window intervals, with overlapping samples counted once. IQR gives the 25th and 75th percentiles (linear interpolation). Clean fraction is accepted divided by candidate windows. The strata overlap; these are descriptive summaries, not independent comparisons or proof that all residual artifacts were removed.')
    d=data['retention'].set_index('subset');names=['readable_126','retained_96','excluded_30']
    rows=[['Participants']+[str(int(d.loc[n,'n'])) for n in names],['Age, mean (SD), years']+[f"{d.loc[n,'age_mean']:.2f} ({d.loc[n,'age_sd_sample']:.2f})" for n in names]]
    for label,col in [('ADHD study group','group__Experimental'),('Control study group','group__Control'),('Male source code','source_sex__Male'),('Female source code','source_sex__Female'),('EPOC X','device_model__Epoc X'),('EPOC+','device_model__Epoc+')]:rows.append([label]+[str(int(d.loc[n,col])) for n in names])
    assert d.loc['readable_126','n']==126 and d.loc['retained_96','n']==96 and d.loc['excluded_30','n']==30
    table('retention_profile','Characteristics of readable, retained, and excluded multimodal records.','tab:retention-profile','lrrr','Characteristic & Readable & Retained & Excluded',rows,'The 126 readable records represent 126 unique participants. Excluded combines 28 measurement-screening and two subsequent EEG-preprocessing exclusions. All participants were aged 6--18 years. Counts and sample SDs are descriptive; no attrition hypothesis test was performed. The comparison is conditional on readable multimodal files, not the full source-study population.')
    rows=[];dn={'unadjusted':'Unadjusted','device':'Device','device_quality':'Device + quality'}
    for _,r in data['pairs'].iterrows():
        assert abs(r.delta_ll-r.tail_union_delta_ll_contribution-r.tail_neither_delta_ll_contribution)<1e-12
        b=int(r.tail_baseline_only_n+r.tail_both_n);e=int(r.tail_augmented_only_n+r.tail_both_n)
        rows.append([dn.get(r.design,r.design),f'{b}/{e}',f'{r.delta_ll:+.4f}',f'{r.tail_union_delta_ll_contribution:+.4f}',f'{r.tail_neither_delta_ll_contribution:+.4f}'])
    table('probability_tails','Saved-probability error contributions for entropy/Hjorth augmentation.','tab:probability-tail-diagnostics','lrrrr',r'Design & Tails B/E & Total $\Delta$LL & Tail contribution & Other contribution',rows,r'B and E denote matched baseline and augmented models. Counts are occurrences among 960 dependent participant--repeat predictions per model, not independent participants. Tail contributions sum paired loss changes for rows meeting the incorrect-tail rule in either model, divided by all 960 rows; the other contribution uses remaining rows with the same denominator. Before rounding, the components sum to total $\Delta$LL. Positive values favor EEG; no rows were excluded or models recalibrated.')
    rows=[];direction={'X_to_Plus':r'X $\to$ Plus','Plus_to_X':r'Plus $\to$ X'};model={'spectral_covariance_selector':'Selector','entropy_hjorth':'Entropy/Hjorth'}
    for _,r in data['transfer'].iterrows():rows.append([direction[r.direction],model[r.model],f'{r.observed_prevalence:.4f}',f'{r.baseline_mean_p:.4f}',f'{r.augmented_mean_p:.4f}'])
    table('transfer_probability','Probability location in the device-transfer comparisons.','tab:transfer-probability-location','llrrr',r'Train $\to$ assess & Procedure & Target fraction & Mean $p_B$ & Mean $p_E$',rows,'B and E denote the source-fitted baseline and EEG-augmented models. Target fraction is the observed ADHD study-group proportion. Mean probability alone does not establish calibration. Paired losses and equal-class summaries are in Table~\\ref{tab:transfer}; no target-label recalibration was applied.')
    (out/'REPORTING_ASSET_VALUES.json').write_text(json.dumps({'input_sha256':{v:hashlib.sha256((root/v).read_bytes()).hexdigest() for v in paths.values()},'generated_tables':['clean_duration','retention_profile','probability_tails','transfer_probability'],'retained_participants':96,'prediction_rows_per_model':960,'figures_changed':False},indent=2),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repository',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.repository,a.output)
