"""Exploratory, saved-prediction audit; never fits a model."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE = REPO / 'audit/preregistered_analysis'
OUT = HERE / 'outputs'
OUT.mkdir(exist_ok=True)
paths = [SOURCE / n for n in
         ['PRIMARY_OUTER_PREDICTIONS.csv', 'PRIMARY_INNER_SELECTION.csv']]
p, inner = [pd.read_csv(path, float_precision='round_trip') for path in paths]
p = p.loc[p.outcome.eq('group')].copy()
inner = inner.loc[inner.outcome.eq('group')].copy()
keys = ['repeat', 'outer_fold']
assert not p.duplicated(['repeat', 'participant_id']).any()
assert p.groupby('repeat').size().eq(96).all()

def loss(y, probability):
    probability = np.clip(np.asarray(probability, dtype=float), 1e-15, 1-1e-15)
    y = np.asarray(y, dtype=float)
    return -(y*np.log(probability) + (1-y)*np.log1p(-probability))

columns = {'baseline':'p_m2', 'selector':'p_adaptive_m3',
           'E0':'p_e0', 'E1':'p_e1', 'E2':'p_e2'}
for name, col in columns.items():
    p[name] = loss(p.y, p[col])
chosen = np.array([p.loc[i, 'p_'+str(rep).lower()]
                   for i, rep in p.m3_representation.items()])
equal_count = int(np.count_nonzero(chosen == p.p_adaptive_m3.to_numpy()))
assert p.groupby(keys).m3_representation.nunique().eq(1).all()
fold = p.groupby(keys)[list(columns)].mean()
fold['chosen'] = p.groupby(keys).m3_representation.first()
frequency = fold.chosen.value_counts().reindex(['E0','E1','E2'], fill_value=0)
weights = frequency / len(fold)
repeat = p.groupby('repeat')[list(columns)].mean()
repeat['independent'] = sum(weights[r]*repeat[r] for r in ['E0','E1','E2'])
fold['best'] = fold[['E0','E1','E2']].idxmin(axis=1)
assert not fold[['E0','E1','E2']].apply(lambda row: row.eq(row.min()).sum()>1, axis=1).any()
observed = int(fold.chosen.eq(fold.best).sum())
expected = float(sum(weights[r]*fold.best.eq(r).sum() for r in weights.index))
inner_best = (inner.loc[inner.model_family.eq('adaptive_M3') & inner.valid.eq(True)]
              .groupby(keys+['representation']).pooled_logloss.min().unstack())
fold = fold.join(inner_best.add_prefix('inner_'), validate='one_to_one')
fold['inner_E2_minus_E0'] = fold.inner_E2-fold.inner_E0
fold['outer_E2_minus_E0'] = fold.E2-fold.E0
rho = float(spearmanr(fold.inner_E2_minus_E0, fold.outer_E2_minus_E0).statistic)
worst_repeat = repeat.selector.idxmax()
omitted = repeat.drop(index=worst_repeat).mean()
means = repeat.mean()
independent_gain = float(means.selector-means.independent)
primary_deficit = float(means.selector-means.baseline)
unique = p.drop_duplicates('participant_id')
prevalence = float(unique.y.mean())
prevalence_loss = float(loss(unique.y, np.repeat(prevalence,len(unique))).mean())
results = [
    ('identical_predictions', equal_count, 960, 0),
    ('group_prediction_rows', len(p), 960, 0),
    *[(name+'_mean_logloss', float(means[name]), value, 4) for name,value in
      [('baseline',.6244),('selector',.6532),('E0',.6199),('E1',.6366),('E2',.6398)]],
    ('E0_selection_fits', int(frequency.E0),22,0),
    ('E2_selection_fits', int(frequency.E2),28,0),
    ('independent_mean_logloss',float(means.independent),.6310,4),
    ('selection_cost',independent_gain,.0222,4),
    ('primary_deficit',primary_deficit,.0288,4),
    *[('selector_worse_than_'+r+'_repeats',
       int(repeat.selector.gt(repeat[r]).sum()),expected_count,0)
      for r,expected_count in [('E0',10),('E1',7),('E2',8),('independent',9)]],
    ('selector_worse_than_baseline_repeats',int(repeat.selector.gt(repeat.baseline).sum()),None,0),
    ('selector_worse_than_all_three_repeats',
     int(repeat[['E0','E1','E2']].lt(repeat.selector, axis=0).all(axis=1).sum()),None,0),
    ('omit_worst_selector',float(omitted.selector),.6435,4),
    ('omit_worst_independent',float(omitted.independent),.6247,4),
    ('omit_worst_baseline',float(omitted.baseline),.6248,4),
    ('selected_outer_best',observed,11,0),
    ('expected_outer_best_under_independence',expected,22.6,1),
    ('spearman_inner_outer_E2_minus_E0',rho,-.69,2),
    ('prevalence_logloss',prevalence_loss,.6825,4),
]
verification = pd.DataFrame(results, columns=['quantity','recomputed','requested','decimals'])
verification['matches_at_requested_precision'] = verification.apply(
    lambda row: (None if pd.isna(row.requested) else
                 round(float(row.recomputed),int(row.decimals)) ==
                 round(float(row.requested),int(row.decimals))), axis=1)
comparison_labels = {
    'selector_worse_than_E0_repeats':'Sabit E0',
    'selector_worse_than_E1_repeats':'Sabit E1',
    'selector_worse_than_E2_repeats':'Sabit E2',
    'selector_worse_than_independent_repeats':'Katmandan bağımsız seçim karışımı',
}
verification['repeat_comparison'] = verification.quantity.map(comparison_labels).fillna('')
verification['worse_repeats'] = verification.apply(
    lambda row: f'{int(row.recomputed)}/10' if row.quantity in comparison_labels else '',axis=1)
verification.to_csv(OUT/'VERIFICATION.csv', index=False, float_format='%.17g')
verification.loc[verification.quantity.isin(comparison_labels),
                 ['repeat_comparison','worse_repeats']].to_csv(OUT/'COMPARISON_REPEATS.csv',index=False)
repeat.to_csv(OUT/'REPEAT_LOGLOSSES.csv', float_format='%.17g')
fold.to_csv(OUT/'FOLD_SELECTION_COMPARISON.csv', float_format='%.17g')
metadata = {
    'date':'2026-11-05', 'analysis_status':'exploratory', 'model_fitting':False,
    'source_sha256':{str(path.relative_to(REPO)):hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in paths},
    'participants':len(unique), 'positive_labels':int(unique.y.sum()),
    'worst_repeat_zero_based':int(worst_repeat),
    'selection_share_of_primary_deficit':independent_gain/primary_deficit,
    'independent_weights':weights.to_dict(),
    'repeat_worse_than_fixed':{r:int(repeat.selector.gt(repeat[r]).sum())
                               for r in ['E0','E1','E2']},
    'user_correction_2026_11_05':
        '9/10 refers to the fold-independent mixture. User confirmed '
        'E0=10/10, E1=7/10, E2=8/10, mixture=9/10. No unresolved discrepancy.',
    'all_requested_numbers_verified':bool(
        verification.matches_at_requested_precision.dropna().all()),
}
(OUT/'METADATA.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print(verification.to_string(index=False))
print(json.dumps(metadata,indent=2))
if not metadata['all_requested_numbers_verified']:
    raise SystemExit('STOP: at least one requested number does not match.')
