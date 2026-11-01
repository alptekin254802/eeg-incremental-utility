# Analysis status and registration crosswalk

This is a current explanatory crosswalk, not an original registration document.
The available saved registration overview identifies OSF 9jp2h, registered on
25 August 2026 and embargoed through 25 August 2028. Its displayed questionnaire contains the design and an explicit Prior knowledge account of the group/diagnosis discrepancy and unanswered contact with the data authors. The primary GROUP target and separate diagnosis-status sensitivity were fixed in that text. Server history and the full registered technical attachment/hash roster have not been independently retrieved; this narrow authentication limit does not mean that the registered design is absent. Reviewer access remains an administrative step.

| Component | Registered description / implemented role | Status |
| --- | --- | --- |
| Cohort and target | Frozen 96-person cohort; GROUP primary; windows form features, not prediction samples | Primary specification retained |
| Baseline | Age, source sex/gender code, behavior, two object-gaze summaries | Primary specification retained |
| EEG library | Regional spectral, expanded spectral, expanded spectral plus covariance; joint C/representation selection | Primary specification retained |
| Evaluation | 10 repeats × 5 outer folds; 5 inner folds; participant-level pairing; training-only transforms | Primary specification retained |
| Score/inference | Mean paired log loss, baseline minus EEG; no primary test or confirmatory interval | Primary specification retained |
| Missingness | Frozen inputs finite; prespecified median safeguard inactive | No new imputation introduced for primary data |
| Diagnosis status | Exact yes/no, newly stratified folds, smaller subset | Prespecified sensitivity; historical prediction archive incomplete |
| Lower modality steps | Demographics, then behavior, then object gaze | Descriptive comparisons |
| Phase-lag, entropy/Hjorth, spectral states | Designed after the primary result | Post hoc, outside primary registration |
| Acquisition, component, selector and phase controls | Planned with previous results known | Post hoc |
| Scalar calibration numerical follow-up | Same diagnostic likelihood solved more robustly; predictions unchanged | Numerical follow-up, not recalibrated prediction |
| Device transfer, learners, learning curves, simulations, adult data | Separate local plans preceding their new fits, with earlier findings known | Exploratory extensions, not original preregistration |
| Review-stage descriptive summaries | Retained duration, probability-error and acquisition reporting from saved records | Post hoc reporting, not predictive model selection |

The available overview and implementation agree on these principal choices.
This is not an exhaustive certification against inaccessible registered technical
attachments. Local plan hashes establish identity of the supplied plans; they
do not independently authenticate a timestamped third-party deposit. The original
plans remain unchanged. Nested tuning addresses within-procedure selection, not
the optimism of designing later representations after seeing results from the
same cohort.

## Later compact learned representation

The EEGNet-like follow-up was planned after the previously reported findings and before its own clinical fits. The dated plan and code lock are in `extensions/eegnet/`. It is exploratory and does not belong to the original registration. Architecture, 50-epoch budget, 300 nested encoders and conditional uncertainty calculation were fixed before fitting; all results are retained.
