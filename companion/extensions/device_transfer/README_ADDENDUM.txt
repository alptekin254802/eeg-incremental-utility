Exploratory device-transfer addendum — 1 November 2026
This analysis was planned after the original study results were known and before the new fits. It is not part of the original preregistration or the archived v1.1.0 release.

Prerequisite: extracted computational release v1.1.0, DOI 10.5281/zenodo.23199244. The frozen feature and outcome files remain in that release. Do not overwrite them.
Software recorded in analysis/results/COMPLETION.json: Python 3.14.3, numpy 2.3.4, pandas 2.3.3, scipy 1.16.3, scikit-learn 1.8.0. Matplotlib and threadpoolctl are used for plotting/thread limits.
Run from this addendum root, replacing RELEASE with that extracted release path:
python -B analysis/code/device_transfer_portable.py --release-root RELEASE --output analysis/replay
python -B analysis/code/verify_saved_results.py --release-root RELEASE --replay analysis/replay
These commands refuse to overwrite completed outputs. The original one-run results remain in analysis/results. One deterministic replay was performed during preparation; all ten CSV files matched byte for byte. The replay output itself is not duplicated in this ZIP.
To rebuild supplement assets from saved results, replacing MANUSCRIPT with the extracted manuscript source root:
python analysis/code/build_transfer_assets.py --manuscript MANUSCRIPT

ANALYSIS_PLAN.txt and audit/PLAN_LOCK.json preserve the pre-fit plan. analysis/code/device_transfer.py is the immutable original script, with its original local path assumptions; use device_transfer_portable.py for reproduction elsewhere. Only input/output path handling differs. Original paths in INPUT_MANIFEST.json record provenance; --release-root overrides them in the verifier.
PREDICTIONS.csv retains all models and target participants. INNER_SCORES.csv and MODEL_SELECTION.csv retain tuning. CONTRASTS_ALL.csv retains every fixed and selected contrast. PRINCIPAL_CONTRASTS.csv contains the four planned main comparisons; GROUP_CONTRASTS.csv retains both target study groups. CONDITIONAL_DRAWS.csv retains all 20,000 paired target bootstrap draws. ABSOLUTE_METRICS.csv supplies absolute loss, AUROC and Brier scores.
Positive loss/Brier differences mean baseline minus EEG; positive AUROC differences mean EEG minus baseline. Equal-class loss is a fixed-prediction reweighting. Conditional ranges omit source fitting/selection uncertainty. No independent clinical validation or causal device effect is claimed.
