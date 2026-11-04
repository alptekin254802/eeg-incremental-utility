"""Check every intentionally changed numerical row against round-1 outputs."""
from pathlib import Path
import pandas as pd
import re

def read(path):return pd.read_csv(path,float_precision='round_trip')

def verify(root:Path,tables:Path):
    ex=root/'exploratory_r2_2026-11-05'
    d=read(ex/'diagnosis/outputs/SUMMARY.csv').set_index('comparison')
    e=read(ex/'eegnet_crossfit/outputs/SUMMARY.csv').set_index('comparison')
    b=read(ex/'recorded_audit/TABLE4_BASELINES.csv').set_index('adjustment').baseline_logloss
    adjust=(tables/'main_adjustment.tex').read_text(encoding='utf-8')
    rows=[line.split(' & ') for line in adjust.splitlines() if line.startswith(
        ('Primary procedure &','Spectral/covariance selector &','Entropy/Hjorth features &'))]
    assert len(rows)==6
    for j,row in enumerate(rows):assert row[1]==f'{b.loc[["none","_D","_DQ"][j//2]]:.4f}'
    assert 'Baseline LL' in adjust
    learner=(tables/'main_learner.tex').read_text(encoding='utf-8')
    assert 'EEGNet' not in learner and '-0.1006' not in learner
    original=(root/'companion/article_assets/tables/main_learner.tex').read_text(encoding='utf-8')
    def numeric_rows(text):
        return [re.findall(r'[-+]?\d+(?:\.\d+)?',line.split('&',2)[2])
                for line in text.splitlines() if line.startswith(('Logistic regression &','RBF SVM &','Random forest &'))
                and 'EEGNet' not in line]
    assert numeric_rows(original)==numeric_rows(learner)
    diagnosis=(tables/'diagnosis.tex').read_text(encoding='utf-8')
    for label,key in [('None','diagnosis_entropy'),('Device + quality','diagnosis_entropy_DQ')]:
        r=d.loc[key]
        expected=label+' & '+' & '.join(f'${r[k]:+.4f}$' for k in ['delta_ll','delta_auroc','delta_brier'])
        assert expected in diagnosis
        assert f'[{r.lower:+.4f},{r.upper:+.4f}]' in diagnosis
        assert int(r.positive_repeats)==7 and '7/10' in diagnosis
    eegnet=(tables/'eegnet_summary.tex').read_text(encoding='utf-8')
    labels=[('Split ensemble','split_ensemble'),('Pooled OOF, fixed $C=1$','pooled_oof_fixed_C1'),
            ('Original embeddings (reference)','original_embedding_reference')]
    for label,key in labels:
        r=e.loc[key]
        expected=label+' & '+f'${r.baseline_logloss:.4f}$ & ${r.augmented_logloss:.4f}$ & ${r.delta_ll:+.4f}$ & $[{r.lower:+.4f},{r.upper:+.4f}]$'
        assert expected in eegnet
        expected=label+' & '+f'${r.delta_auroc:+.4f}$ & ${r.delta_brier:+.4f}$ & '+r'\multicolumn{2}{r}{'+f'{int(r.positive_repeats)}/10'+r'}'
        assert expected in eegnet
    adult=(tables/'adult_extension.tex').read_text(encoding='utf-8')
    assert 'Broader' not in adult and 'broader' not in adult and 'Task/eye' not in adult
    assert 'All non-EEG' in adult and 'Task + eye' in adult
    return dict(status='PASS',baseline_adjustment_rows=6,learner_rows=6,
                diagnosis_new_rows=2,crossfit_and_reference_rows=3,
                source='Frozen round-1 output CSVs; no fitting or new resampling')
