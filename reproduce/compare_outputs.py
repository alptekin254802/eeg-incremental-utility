"""Compare newly computed E5/post-hoc predictions and metrics to reference results."""
from pathlib import Path
import argparse
import json
import os
import numpy as np
import pandas as pd


def compare(new, old, keys, exact=()):
    a, b = pd.read_csv(new), pd.read_csv(old)
    assert set(a.columns) == set(b.columns), (new.name, 'columns differ')
    m = a.merge(b, on=keys, suffixes=('_new', '_reference'), validate='one_to_one', how='outer', indicator=True)
    assert len(m) == len(a) == len(b) and (m._merge == 'both').all(), (new.name, 'rows differ')
    errors = {}
    for c in a.columns:
        if c in keys:
            continue
        x, y = m[c+'_new'], m[c+'_reference']
        if pd.api.types.is_numeric_dtype(x) and c not in exact:
            assert np.array_equal(x.isna(), y.isna()), (new.name, c, 'missingness')
            finite = x.notna()
            error = float(np.max(np.abs(x[finite].astype(float)-y[finite].astype(float)))) if finite.any() else 0.
            assert error <= 1e-7, (new.name, c, error)
            errors[c] = error
        else:
            assert (x.fillna('') == y.fillna('')).all(), (new.name, c, 'exact comparison')
    return dict(file=new.name, rows=len(a), max_errors=errors, passed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['e5', 'posthoc'])
    args = parser.parse_args()
    release = Path(os.environ['EEG_RELEASE_ROOT'])
    run = Path(os.environ['EEG_RUN_ROOT'])
    if args.stage == 'e5':
        new, old = run/'e5', release/'results/e5'
        specs = [('OUTER_PREDICTIONS.csv', ['family','repeat','outer_fold','participant_id'], ['selected_C','participant_index','y']),
                 ('REPEAT_RESULTS.csv', ['family','repeat'], ['n'])]
    else:
        new, old = run/'posthoc/results', release/'expected/posthoc/results'
        specs = [('outer_predictions.csv', ['model','repeat','outer_fold','participant_id'], ['selected_C','y']),
                 ('repeat_results.csv', ['comparison','repeat'], []),
                 ('comparison_summary.csv', ['comparison'], [])]
    checks = [compare(new/name, old/name, keys, exact) for name, keys, exact in specs]
    (new/'REFERENCE_CHECKS.json').write_text(json.dumps(dict(stage=args.stage, checks=checks, passed=True), indent=2)+'\n')
    print(json.dumps(dict(stage=args.stage, checks=checks, passed=True), indent=2))


if __name__ == '__main__':
    main()
