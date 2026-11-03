"""Build the two supplementary tables from complete saved sensitivity results."""
import argparse,json,hashlib
from pathlib import Path
import pandas as pd
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    s=pd.read_csv(a.results/'summary/SENSITIVITY_SUMMARY.csv');o=pd.read_csv(a.results/'offset/CONTRASTS.csv')
    names={'age_primary':'Age spline: primary selector','age_entropy':'Age spline: entropy/Hjorth','age_entropy_DQ':'Age spline: entropy/Hjorth + DQ','bandwidth_entropy':'Low-pass: entropy/Hjorth','bandwidth_entropy_DQ':'Low-pass: entropy/Hjorth + DQ'}
    lines=[r'\begin{table}[htbp]',r'\centering',r'\caption{Targeted age and bandwidth sensitivities. Positive log-loss differences favor EEG augmentation.}',r'\label{tab:targeted}',r'\begin{tabular}{@{}p{4.8cm}rrlr@{}}',r'\toprule',r'Comparison & Original & Sensitivity & Conditional range & $+$/10 \\',r'\midrule']
    for r in s.to_dict('records'):
        lines.append(f"{names[r['comparison']]} & ${r['reference_delta_ll']:+.4f}$ & ${r['delta_ll']:+.4f}$ & $[{r['lower']:+.4f},\\,{r['upper']:+.4f}]$ & {r['positive_repeats']} "+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\par\smallskip\begin{minipage}{\textwidth}',r'Original: corresponding linear-age, 1--40~Hz analysis. DQ: device, clean-window fraction, and interpolated-channel count in both models. The age and low-pass changes were evaluated separately. Low-pass refers to an additional zero-phase 30~Hz filter applied to the original preprocessed segments. Ranges are participant-resampling percentiles conditional on fixed predictions; they omit training and selection uncertainty. $+$/10 counts repeats with positive increments, not independent replications.',r'\end{minipage}',r'\end{table}']
    (a.out/'supp_targeted.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    lines=[r'\begin{table}[htbp]',r'\centering',r'\caption{Oracle prior-odds sensitivity of the saved device-transfer comparisons.}',r'\label{tab:offset}',r'\begin{tabular}{@{}llrrrr@{}}',r'\toprule',r'& & \multicolumn{2}{c}{Observed composition} & \multicolumn{2}{c}{Equal-class mean} \\',r'\cmidrule(lr){3-4}\cmidrule(l){5-6}',r'Direction & EEG procedure & Original & Offset & Original & Offset \\',r'\midrule']
    for direction in ['Plus_to_X','X_to_Plus']:
        for model in ['spectral_covariance_selector','entropy_hjorth']:
            g=o[(o.direction==direction)&(o.model==model)].set_index('correction');b=g.loc['original'];c=g.loc['oracle_prior_odds'];di=r'EPOC+ $\to$ X' if direction=='Plus_to_X' else r'EPOC X $\to$ +';mo='Selector' if model.startswith('spectral') else 'Entropy/Hjorth'
            lines.append(f"{di} & {mo} & ${b.delta_ll:+.4f}$ & ${c.delta_ll:+.4f}$ & ${b.equal_class_delta_ll:+.4f}$ & ${c.equal_class_delta_ll:+.4f}$ "+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\par\smallskip\begin{minipage}{\textwidth}',r'Entries are baseline-minus-augmented log-loss differences. Offset adds the same source-to-target prior-log-odds change to both models. It uses target study-group labels and is an oracle diagnostic, not a deployable predictor. All offset conditional ranges, both observed-composition and equal-class, include zero. Complete probabilities, absolute losses, secondary metrics, group-specific differences, and conditional draws accompany the table.',r'\end{minipage}',r'\end{table}']
    (a.out/'supp_offset.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    files=['summary/SENSITIVITY_SUMMARY.csv','offset/CONTRASTS.csv']
    values=dict(inputs={n:hashlib.sha256((a.results/n).read_bytes()).hexdigest() for n in files},sensitivity=s.to_dict('records'),offset=o.to_dict('records'))
    (a.out/'TARGETED_TABLE_VALUES.json').write_text(json.dumps(values,indent=2)+'\n')
    print('Two supplementary tables generated from all five fit and four offset comparisons.')
if __name__=='__main__':main()
