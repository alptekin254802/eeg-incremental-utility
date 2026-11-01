"""Report agreement of predictions and model selections separately from inner scores."""
import numpy as np


def classify_replay(model_checks, paired_scores, diagnosis_errors, failures, strict=False):
    difference = np.abs(paired_scores.pooled_logloss_new - paired_scores.pooled_logloss_old)
    valid_equal = bool((paired_scores.valid_new == paired_scores.valid_old).all())
    ranks_equal = bool((paired_scores.rank_new.fillna(0) == paired_scores.rank_old.fillna(0)).all())
    finite = bool(np.isfinite(difference).all())
    mismatches = paired_scores.loc[difference > 1e-7].copy()
    mismatches['absolute_error'] = difference[difference > 1e-7]
    # Report the known diagnostic difference while keeping the original tolerance.
    known_scope = bool(((mismatches.C == 100) & (mismatches.rank_new > 1)
                       & (mismatches.rank_old > 1)).all())
    results_match = bool(all(r['passed'] for r in model_checks) and
                         max(diagnosis_errors.values()) <= 1e-7 and not failures)
    intermediate_match = bool(finite and valid_equal and ranks_equal and mismatches.empty)
    warning_only = bool(results_match and finite and valid_equal and ranks_equal and known_scope)
    strict_passed = bool(results_match and intermediate_match)
    accepted = strict_passed if strict else bool(strict_passed or warning_only)
    status = ('REPRODUCED' if strict_passed else
              'RESULTS_REPRODUCED_WITH_INTERMEDIATE_DIFFERENCES' if warning_only else 'FAILED')
    return dict(status=status, predictive_results_passed=results_match,
                intermediate_scores_passed=intermediate_match, passed=strict_passed,
                command_completed=accepted, strict_intermediates=bool(strict),
                validity_equal=valid_equal, ranks_equal=ranks_equal,
                intermediate_mismatch_count=len(mismatches)), mismatches
