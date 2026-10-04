# Numerical reproduction limits

A recorded fresh-input replay of the analysis reproduced the selected models and outer predictions within the stated comparison tolerances. It did not reproduce every intermediate diagnostic exactly. These distinctions matter when comparing a new run with the supplied reference outputs.

- Some inner-validation scores for the unselected regularization candidate `C=100` exceeded the `1e-7` comparison threshold. Selected regularization values and outer predictions agreed. The tolerance is retained; differences in intermediate scores should be reported rather than silently accepted.
- Scalar calibration estimation can terminate with precision-loss warnings. The set of successful fits differed in the replay; estimates for fits successful in both runs agreed, and the reliability-bin results agreed. The published scalar calibration results remain the archived results, with their reported failures.
- For participants excluded before complete feature extraction, some diagnostic fields used zero in one record and an unavailable value in another. Eligibility decisions were unchanged. Missing diagnostic values should not be interpreted as measured zeros.
- Historical participant predictions and selected regularization values for the diagnosis-status sensitivity were not included in the original result archive. Its saved repeat-level metrics provide the released numerical reference; a new run can produce participant predictions, but their historical identity cannot be checked against an absent reference.

The short identity checks, tests and figure/table commands do not establish a new full raw-data replay. Raw-input, saved-input and saved-result commands are distinguished in the reproduction guide. Keep the prespecified comparison tolerances when assessing a new environment.
