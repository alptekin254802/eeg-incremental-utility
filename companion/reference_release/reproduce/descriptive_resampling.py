"""Resample participant losses while holding out-of-fold predictions fixed.

The paired participant mean losses are resampled with all models, partitions,
selection decisions and predictions held fixed. These are not confidence
intervals or hypothesis tests of the complete learning procedure.
"""
from pathlib import Path
import hashlib
import json
import os
import numpy as np
import pandas as pd


def main():
    base = Path(os.environ['EEG_RELEASE_ROOT'])
    out = Path(os.environ['EEG_RUN_ROOT']) / 'descriptive_resampling'
    out.mkdir(parents=True, exist_ok=True)
    source = base / 'expected/posthoc/results/participant_loss_diagnostics.csv'
    participants = pd.read_csv(source)
    order = pd.read_csv(base / 'audit/preregistered_analysis/PRIMARY_OUTER_PREDICTIONS.csv')
    ids = order[order.repeat == 0].sort_values('participant_index').participant_id.tolist()
    assert len(ids) == len(set(ids)) == 96
    columns = []
    for model in ['original_adaptive', 'E4']:
        group = participants[participants.model == model].set_index('participant_id')
        assert group.index.is_unique and set(group.index) == set(ids)
        columns.append(group.loc[ids, 'mean_delta_ll'].to_numpy(float))
    values = np.column_stack(columns)
    draws = []
    for b in range(5000):
        payload = f'20260824125|group|fixed_oof|{b}'.encode('utf-8')
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:4], 'little')
        indices = np.random.Generator(np.random.PCG64(seed)).integers(0, 96, size=96)
        draws.append(values[indices].mean(axis=0))
    draws = np.asarray(draws)
    ranges = np.quantile(draws, [.025, .975], axis=0, method='linear')
    result = pd.DataFrame([dict(model=model, mean_delta_ll=float(values[:, i].mean()),
                               lower=float(ranges[0, i]), upper=float(ranges[1, i]),
                               participants=96, resamples=5000)
                           for i, model in enumerate(['original_adaptive', 'E4'])])
    result.to_csv(out / 'FIXED_OOF_RANGES.csv', index=False, float_format='%.17g')
    (out / 'SPECIFICATION.json').write_text(json.dumps(dict(
        analysis_status='post hoc', label='conditional descriptive fixed-OOF resampling range',
        resamples=5000, percentile_method='linear', quantiles=[.025, .975],
        root_seed=20260824125, child_seed='first four SHA-256 bytes of root|group|fixed_oof|b, little endian',
        participant_order='primary prediction participant_index, repeat 0',
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        inferential_claims=False, predictive_models_refitted=False), indent=2) + '\n')
    print(result.to_string(index=False))


if __name__ == '__main__':
    main()
