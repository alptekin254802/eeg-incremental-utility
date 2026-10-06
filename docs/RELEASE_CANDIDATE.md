# Version 1.1.0

The `figures` and `tables` commands now reproduce the figures and tables in the
current article from the saved results. The code is in `display_assets/`, and generated files are written outside the repository; the full article and supplement are maintained separately.

## What changed

- **Figures and tables.** The commands generate main Figures 1–3, supplementary Figures S1–S3, main Tables 1–3 and supplementary Tables S1–S8 under `<output>/display_assets/`. They read the supplied results directly. The additional calibration and resampling results are in `results/numerical_followups/`.
- **Model checks.** Verification reports whether predictions and model selections agree, and separately checks intermediate scores against the original `1e-7` tolerance. When differences are limited to the documented unselected `C=100` scores, the command finishes with a warning and writes the affected rows to a CSV file. Those scores still fail the numerical check. Use `--strict-intermediates` to make the command exit with an error in this case too.
- **E5 and post-hoc comparisons.** These commands now compare regenerated predictions and summary metrics with the supplied reference results. Selected regularization values must match exactly.
- **Calibration.** `calibration-numerical` estimates the same unpenalized calibration model from saved probabilities using damped Newton iteration, then checks the solutions with an independently initialized root solver. We retain the original BFGS results. This post-hoc check does not refit the predictive models or change their probabilities.
- **Descriptive resampling.** `descriptive-resampling` resamples paired participant mean losses while keeping all model fits and out-of-fold predictions fixed. The resulting post-hoc ranges describe variation conditional on those fits; they do not measure uncertainty from repeating the complete fitting procedure or provide significance tests.

The raw measurements, features, folds, predictive estimators, archived predictions
and original result tables are unchanged. Supplementary Sections S1–S3 describe the existing gender coding, signal segmentation, aggregation order and feature variance conventions.

## Citation

Use the Zenodo concept DOI [10.5281/zenodo.23146468](https://doi.org/10.5281/zenodo.23146468)
to cite the software across all published versions.
