"""Generate manuscript table fragments from numerical results."""
from pathlib import Path
import csv
import hashlib
import json
import shutil

import os
ROOT = Path(os.environ['EEG_RELEASE_ROOT'])
HERE = Path(os.environ['EEG_RUN_ROOT'])
SOURCE = HERE / 'posthoc'
CURRENT = HERE
TABLES = CURRENT / 'tables'
DATA = CURRENT / 'supplementary_data'
for directory in (TABLES, DATA, CURRENT / 'figures'):
    directory.mkdir(exist_ok=True)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))

rows = read_rows(SOURCE / 'results/comparison_summary.csv')
lookup = {r['comparison']: r for r in rows}
assert len(rows) == len(lookup) == 35
surrogates = read_rows(SOURCE / 'signal_controls/comparison_summary.csv')
surrogate_lookup = {r['comparison']: r for r in surrogates}
assert len(surrogates) == len(surrogate_lookup) == 40

labels = {
    'M2': 'M2', 'M2_D': 'M2+D', 'M2_DQ': 'M2+DQ',
    'E4': 'M2+E4', 'E4_D': 'M2+E4+D', 'E4_DQ': 'M2+E4+DQ',
    'E0_relative9': 'M2+relative spectrum (9)', 'E0_PCA9': 'M2+E0 PCA (9)',
    'original_adaptive': 'Adaptive M3', 'adaptive_D': 'Adaptive M3+D',
    'adaptive_DQ': 'Adaptive M3+DQ', 'optout': 'Opt-out policy',
}
for family in ('E0', 'E1', 'E2'):
    labels['original_' + family] = 'M2+' + family
    for adjustment in ('D', 'DQ'):
        labels[family + '_' + adjustment] = f'M2+{family}+{adjustment}'
for block, short in [('perm_entropy', 'PE'), ('hjorth_mobility', 'HM'), ('hjorth_complexity', 'HC')]:
    labels['E4_only_' + block] = f'M2+{short} only'
    labels['E4_without_' + block] = f'M2+E4 without {short}'

def signed(value):
    return f'${float(value):+.4f}$'

def metrics(row):
    return ' & '.join(signed(row[k]) for k in ('delta_ll', 'delta_auroc', 'delta_brier'))

def write(name, body):
    (TABLES / name).write_text('% Generated from the reported numerical results.\n' + body + '\n', encoding='utf-8')

main_comparisons = [
    ('M2_vs_E4', 'Full E4'),
    ('M2_vs_E4_only_perm_entropy', 'Permutation entropy only'),
    ('M2_D_vs_E4_D', 'E4, device adjustment'),
    ('M2_DQ_vs_E4_DQ', 'E4, device and quality adjustment'),
    ('M2_D_vs_adaptive_D', 'Adaptive E0--E2, device adjustment'),
    ('M2_DQ_vs_adaptive_DQ', 'Adaptive E0--E2, device and quality adjustment'),
    ('M2_vs_E0_relative9', 'Nine relative-power predictors'),
    ('M2_vs_E0_PCA9', 'Nine E0 principal components'),
    ('M2_vs_optout', 'Selection with EEG opt-out'),
]
body = r'''\begin{table*}[t]
\centering
\caption{Selected post-hoc explanatory comparisons with matched baselines.}
\label{tab:posthoc-controls}
\small
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.12}
\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}Xrrrr@{}}
\toprule
Evaluated EEG addition or policy & $\Delta_{\mathrm{LL}}$ & $\Delta_{\mathrm{AUROC}}$ & $\Delta_{\mathrm{Brier}}$ & Positive repeats \\
\midrule
'''
for key, label in main_comparisons:
    row = lookup[key]
    body += f'{label} & {metrics(row)} & {row["positive_repeats"]}/10 ' + r'\\' + '\n'
body += r'''\bottomrule
\end{tabularx}
\begin{minipage}{\textwidth}
\footnotesize\textit{Note.} Positive differences favor the evaluated model. Device-adjusted and device/quality-adjusted comparisons use M2 with the same added covariates as baseline (mean LL 0.6094 and 0.6180, respectively); all other rows use M2 (0.6244). Quality covariates are clean-window fraction and interpolated-channel count. Positive repeats count favorable log-loss differences among ten correlated repeats, not independent successes. The complete 35-comparison set is reported in Supplementary Table~S7.
\end{minipage}
\end{table*}'''
write('posthoc_main_summary.tex', body)

body = r'''\begingroup
\small
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.12}
\begin{longtable}{@{}p{0.36\textwidth}rrrrrr@{}}
\caption{All comparisons in the post-hoc explanatory package, including archived reference results.}\label{tab:s7-comparisons}\\
\toprule
Baseline $\to$ evaluated model & LL$_0$ & LL$_1$ & $\Delta_{\mathrm{LL}}$ & $\Delta_{\mathrm{AUROC}}$ & $\Delta_{\mathrm{Brier}}$ & $R_+$ \\
\midrule
\endfirsthead
\multicolumn{7}{l}{\tablename\ \thetable\ (continued)}\\
\toprule
Baseline $\to$ evaluated model & LL$_0$ & LL$_1$ & $\Delta_{\mathrm{LL}}$ & $\Delta_{\mathrm{AUROC}}$ & $\Delta_{\mathrm{Brier}}$ & $R_+$ \\
\midrule
\endhead
\midrule
\multicolumn{7}{r}{Continued on next page}\\
\endfoot
\bottomrule
\endlastfoot
'''
for row in rows:
    description = labels[row['baseline']] + r' $\to$ ' + labels[row['augmented']]
    body += f'{description} & {float(row["baseline_logloss"]):.4f} & {float(row["augmented_logloss"]):.4f} & {metrics(row)} & {row["positive_repeats"]} ' + r'\\' + '\n'
body += r'''\end{longtable}
\footnotesize LL$_0$ and LL$_1$ are mean baseline and evaluated log loss. $R_+$ is the number of positive log-loss differences among ten correlated repeats. D: device; DQ: device plus clean-window fraction and interpolated-channel count; PE: permutation entropy; HM: Hjorth mobility; HC: Hjorth complexity. All EEG models include M2. The arrow specifies the comparison direction, including contrasts in which full E4 is the baseline. Adaptive M3 and fixed E0--E2 without adjustment reproduce archived reference comparisons; the new explanatory analyses remain post-hoc. No row was selected or omitted according to performance.
\endgroup'''
write('posthoc_all_comparisons.tex', body)

body = r'''\begin{table}[p]
\centering
\small
\setlength{\tabcolsep}{6pt}
\renewcommand{\arraystretch}{1.18}
\caption{All 20 Fourier-phase controls and both comparison directions.}
\label{tab:s8-surrogates}
\begin{tabular}{rrrrrrrrrr}
\toprule
& & \multicolumn{4}{c}{Control relative to M2} & \multicolumn{4}{c}{Observed E4 relative to control} \\
\cmidrule(lr){3-6}\cmidrule(lr){7-10}
Index & Control LL & $\Delta_{\mathrm{LL}}$ & $\Delta_{\mathrm{AUROC}}$ & $\Delta_{\mathrm{Brier}}$ & $R_+$ & $\Delta_{\mathrm{LL}}$ & $\Delta_{\mathrm{AUROC}}$ & $\Delta_{\mathrm{Brier}}$ & $R_+$ \\
\midrule
'''
for index in range(20):
    control = surrogate_lookup[f'M2_vs_surrogate_{index:02d}']
    observed = surrogate_lookup[f'surrogate_{index:02d}_vs_observed_E4']
    body += f'{index:02d} & {float(control["augmented_logloss"]):.4f} & {metrics(control)} & {control["positive_repeats"]} & {metrics(observed)} & {observed["positive_repeats"]} ' + r'\\' + '\n'
body += r'''\bottomrule
\end{tabular}
\par\medskip
\begin{minipage}{0.95\linewidth}
\footnotesize Positive differences favor the control in the left block and observed E4 in the right block. M2 LL = 0.6244; observed E4 LL = 0.5977. $R_+$ counts positive log-loss differences among ten correlated CV repeats. Indices identify datasets generated by a fixed seed family, not an ordered performance ranking. The 20 controls share participants and do not constitute independent replications. These 40 comparisons are descriptive, without a surrogate $p$ value or multiplicity-adjusted significance claim. Full-precision and repeat-specific values accompany the manuscript.
\end{minipage}
\end{table}'''
write('posthoc_all_surrogates.tex', body)
