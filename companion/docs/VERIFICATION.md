# Verification scope

The original procedures and preceding extensions were checked in separate copies of the repository, using the existing
Python 3.14.3 environment with the pinned scientific dependencies. A clean file
copy is not a fresh operating-system or dependency installation. Original source
files and directory inventories were checked before and after commands.

| Level | Executed checks | What it does not establish |
| --- | --- | --- |
| File identity | Candidate/nested manifests, frozen feature and plan identities, six adult source files, and historical code-hash mapping | Immutable raw-export identity or independent registration history |
| Saved-result arithmetic | Primary and fixed-family predictions, 35 post-hoc contrasts, 40 phase-control contrasts, learner/training-size and simulation results, all six adult pairs and transfer results | New training runs or upstream raw-signal correctness |
| Conditional resampling | Original fixed-OOF endpoints; all 20,000 transfer and 60,000 adult saved contrast rows replayed; a separate tool checks 20/30 selected comparison-draw rows | Training, selection or new-population uncertainty |
| Bounded predictive refits | One original baseline/entropy-Hjorth outer-fold pair (52 fits); one full-size unadjusted empirical-extension task, repeat 0/fold 0 (365 fits, zero retries, exact compared outputs) | Full original or extension model campaigns |
| Figure/table regeneration | All 11 figure PDFs have identical rendered pixels to supplied article figures; seven utility summary CSVs match; current table references and reporting tables checked | New fitted results or raw-data validation |
| Portability contracts | 11 focused tests pass; one native symlink test skipped because the Windows account cannot create symlinks | Native execution on a POSIX operating system |

Selected replay comparisons preserve the original numerical tolerances and model
choices. The device/adult draw replay evaluates conditional distributions only.
Scalar calibration diagnostics are recomputed from saved probabilities; no test
labels are used to recalibrate predictive models. The probability-tail reporting
tool performs no predictive or calibration fit.

The original display generator emits one historical section pointer in the
all-comparisons table. The new wrapper explicitly adapts that **generated output**
to the current cross-reference and records the change. Original code and its
display-value record remain unchanged. The probability-tail table states that its
components sum before rounding; rounded cells may differ by 0.0001.

No complete empirical-extension, simulation, adult or transfer refit was run
during this packaging check. No full original raw-data end-to-end replay, new training bootstrap,
native POSIX run, or fresh dependency installation for those earlier procedures was performed. The commands
for those broader workflows are provided with their requirements, without claiming
that their availability is evidence of execution.

Historical diagnosis-sensitivity participant predictions and regularization
choices are missing from the original archive. The retained repeat metrics can be
compared, but absent historical files cannot receive a byte-identity check. The
raw BALLADEER export/version and complete registered technical attachment history
remain separate provenance limits. Negative findings and all planned exploratory
families remain in the package.

## Compact learned-representation follow-up

The new comparison was run in full: all 300 nested EEGNet-like encoders, 2,600
logistic heads, and 960 paired outer predictions. The preprocessing reconstructed
all 96 participants from raw EEG while retaining frozen baseline features, labels,
and folds; this is not a complete rerun of the original behavioral/gaze pipeline.
A fresh independent window reconstruction produced identical raw/cache identities.
Preprocessing used the original Python 3.14.3/NumPy 2.3.4 environment; neural
training used a newly installed, separately pinned Python 3.12.14/PyTorch environment.
The original preprocessing code was not modified for the older training NumPy.

All nested memberships, candidate scores and saved logistic-head probabilities
were independently re-scored. The maximum baseline prediction discrepancy from
the original archive was 1.97e-11, below the locked 1e-7 tolerance. A selected
repeat-0/outer-fold-0 replay refitted six encoders and 52 heads from fresh windows:
all checkpoint files, 126 state tensors, 24 embedding/normalization arrays,
training curves, candidate scores, and held-out probabilities matched exactly.
This selected replay is not a second full 300-encoder campaign.

The 10,000 class-stratified participant draws condition on fixed fitted predictions;
they are not training bootstraps. Synthetic checks establish held-out-signal
invariance of fitted training states and normalization, assessment-order invariance,
and no batch-normalization update during assessment. One architecture and one
initialization per partition were used; no optimization or architecture adequacy
claim follows from a passing execution check.

Detailed evidence is retained in `extensions/eegnet/reports/VERIFICATION.json`,
`FRESH_WINDOW_IDENTITY.json`, `SYNTHETIC_TESTS.json`, and
`SELECTED_REPLAY_VERIFICATION.json` under the same extension. The final article map
contains 12 figures and 22 table-source files; regeneration and clean-source PDF
checks are separate from model training.
