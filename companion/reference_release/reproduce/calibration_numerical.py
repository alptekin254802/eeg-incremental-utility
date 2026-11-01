"""Fit the original unpenalized calibration model with an alternative solver.

Uses saved primary out-of-fold probabilities; no predictive model is refitted.
The original BFGS outputs remain in audit/preregistered_analysis.
"""
from pathlib import Path
import hashlib
import json
import os
import numpy as np
import pandas as pd
from scipy.optimize import brentq, root
from scipy.special import expit


def solve_calibration(y, probability):
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(probability, dtype=float), 1e-8, 1 - 1e-8)
    z = np.log(p / (1 - p))
    x = np.column_stack([np.ones(len(z)), z])
    if np.linalg.matrix_rank(x) != 2 or len(np.unique(y)) != 2:
        raise ValueError('Calibration requires a full-rank design and both outcomes')

    def objective(b):
        eta = x @ b
        return float(np.sum(np.logaddexp(0, eta) - y * eta))

    def score(b):
        return x.T @ (expit(x @ b) - y)

    def hessian(b):
        q = expit(x @ b)
        return x.T @ ((q * (1 - q))[:, None] * x)

    b = np.array([0., 1.])
    for iteration in range(100):
        g = score(b)
        if np.max(np.abs(g)) <= 1e-9:
            break
        step = np.linalg.solve(hessian(b), g)
        scale = 1.
        old = objective(b)
        while scale > 2.**-40:
            candidate = b - scale * step
            if objective(candidate) <= old - 1e-4 * scale * float(g @ step) + 1e-13:
                break
            scale *= .5
        b = candidate
    gradient = float(np.max(np.abs(score(b))))
    minimum_eigenvalue = float(np.linalg.eigvalsh(hessian(b)).min())
    # Check the same score equations with a second solver and a different starting point.
    alternative = root(score, np.array([0., 0.]), jac=hessian, method='hybr', options={'xtol': 1e-11})
    agreement = float(np.max(np.abs(b - alternative.x)))
    if not (np.isfinite(b).all() and gradient <= 1e-8 and minimum_eigenvalue > 1e-10
            and agreement <= 1e-7 and np.max(np.abs(score(alternative.x))) <= 1e-8):
        raise RuntimeError('Calibration solution did not meet numerical checks')
    citl = brentq(lambda a: np.sum(y - expit(a + z)), -50., 50.,
                  xtol=1e-12, rtol=4*np.finfo(float).eps, maxiter=1000)
    return dict(intercept=float(b[0]), slope=float(b[1]), citl=float(citl),
                gradient_max=gradient, hessian_min_eigenvalue=minimum_eigenvalue,
                alternative_parameter_max_error=agreement, iterations=iteration + 1,
                negative_log_likelihood=objective(b))


def main():
    base = Path(os.environ['EEG_RELEASE_ROOT'])
    out = Path(os.environ['EEG_RUN_ROOT']) / 'calibration_numerical'
    out.mkdir(parents=True, exist_ok=True)
    source = base / 'audit/preregistered_analysis/PRIMARY_OUTER_PREDICTIONS.csv'
    original = pd.read_csv(base / 'audit/preregistered_analysis/CALIBRATION_RESULTS.csv')
    original = original[original.record_type == 'scalar'].set_index(['model', 'repeat'])
    predictions = pd.read_csv(source).sort_values(['repeat', 'participant_index'])
    rows = []
    for model, column in [('M2', 'p_m2'), ('adaptive_M3', 'p_adaptive_m3')]:
        for repeat, group in predictions.groupby('repeat'):
            prior = original.loc[(model, repeat)]
            row = dict(model=model, repeat=int(repeat), original_failure=bool(pd.notna(prior.failure_code)))
            row.update(solve_calibration(group.y, group[column]))
            if not row['original_failure']:
                row['original_parameter_max_error'] = float(max(abs(row[k] - prior[k]) for k in ['intercept', 'slope', 'citl']))
                assert row['original_parameter_max_error'] <= 1e-7
            rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(out / 'CALIBRATION_NUMERICAL.csv', index=False, float_format='%.17g')
    checks = dict(status='NUMERICAL_FOLLOWUP_VERIFIED', fits=len(result),
                  predictive_models_refitted=False, original_results_replaced=False,
                  prediction_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  max_gradient=float(result.gradient_max.max()),
                  max_alternative_parameter_error=float(result.alternative_parameter_max_error.max()))
    (out / 'CHECKS.json').write_text(json.dumps(checks, indent=2) + '\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
