# Current article display map

The map refers to the reviewed manuscript candidate. Figure numbers are assigned
by the manuscript, not by historical console messages in preserved generators.
Device transfer and the adult comparison are **main Figures 4 and 5**.
The original archived release has a smaller scope; its old numbering does not
override this map. All figures use saved results; no conceptual illustration is
substituted for an empirical result.

## Figures

| Current location | File under `article_assets/figures/` | Input family / generator |
| --- | --- | --- |
| Main Figure 1 | `repeat_increments.pdf` | Original repeated predictions; `reference_release/display_assets/figure_source/build_assets.py` |
| Main Figure 2 | `acquisition_phase_controls.pdf` | Original acquisition and phase-control comparisons; same generator |
| Main Figure 3 | `component_increments.pdf` | Original post-hoc feature-block comparisons; same generator |
| Main Figure 4 | `device_transfer.pdf` | `extensions/device_transfer/analysis/results/`; `analysis/code/build_transfer_assets.py` |
| Main Figure 5 | `independent_adult_increment.pdf` | `extensions/utility/results/external/`; `code/summarize_extension.py` |
| Supplement Figure S1 | `calibration.pdf` | Original reliability bins/numerical calibration; original display generator |
| Supplement Figure S2 | `participant_changes.pdf` | Original saved participant changes; original display generator |
| Supplement Figure S3 | `coefficients.pdf` | Original saved coefficients; original display generator |
| Supplement Figure S4 | `model_family_sensitivity.pdf` | Utility empirical predictions/reports; utility summary generator |
| Supplement Figure S5 | `learning_curves.pdf` | Utility empirical predictions/reports; utility summary generator |
| Supplement Figure S6 | `controlled_simulations.pdf` | All utility simulation results/reports; utility summary generator |
| Supplement Figure S7 | `eegnet_comparison.pdf` | `extensions/eegnet/results/`; `code/summarize.py` |

## Tables

| Current location | Table source / content | Source family |
| --- | --- | --- |
| Main Table 1 | `main_cohort.tex` | Original linkage and preprocessing QC |
| Main Table 2 | `main_performance.tex` | Original primary and later-family summaries |
| Main Table 3 | `main_adjustment.tex` | Original acquisition-adjusted paired results |
| Main Table 4 | `main_learner.tex` | Utility empirical summary plus the later EEGNet-like paired comparison |
| Supplement Table S1 | `cohort.tex` | Original 126→98→96 flow and diagnosis subset |
| Supplement Table S2 | Identifier/dimension crosswalk in Methods; `identifier_crosswalk.tex` is the generator counterpart | Fixed representation definitions, not an estimated result |
| Supplement Table S3 | `full_performance.tex` | Original complete performance |
| Supplement Table S4 | `diagnosis.tex` | Original diagnosis-sensitivity repeat metrics |
| Supplement Table S5 | `calibration.tex` | Original numerical calibration follow-up |
| Supplement Table S6 | `all_comparisons.tex` | All 35 original post-hoc contrasts |
| Supplement Table S7 | `phase_comparisons.tex` | All 40 original Fourier-control contrasts |
| Supplement Table S8 | `fixed_oof_ranges.tex` | Original conditional fixed-prediction ranges |
| Supplement Table S9 | `supp_transfer.tex` | Transfer contrasts and conditional draws |
| Supplement Table S10 | `model_extension.tex` | Utility model-family results |
| Supplement Table S11 | `learning_extension.tex` | Utility training-size results |
| Supplement Table S12 | `simulation_extension.tex` | All scenario/learner/size summaries |
| Supplement Table S13 | `adult_extension.tex` | All six adult paired comparisons |
| Supplement Table S14 | `clean_duration.tex` | Recorded union duration and clean-window fraction |
| Supplement Table S15 | `retention_profile.tex` | Existing-record attrition profile |
| Supplement Table S16 | `probability_tails.tex` | Saved-prediction error contributions |
| Supplement Table S17 | `transfer_probability.tex` | Transfer prevalence and mean probabilities |

Original tables use the original display generator; utility tables use
`extensions/utility/code/build_extension_assets.py`. Tables S14–S17 use
`tools/build_reporting_assets.py`. New reporting inputs retain their own plan,
source hashes and verification scope. `reproduce.py assets` stages legacy writers
outside the repository and applies the current map. The manuscript source package
contains the final captions, labels and cross-document links; it remains separate
from this computational companion.

Supplement Table S18 is `eegnet_summary.tex`, generated from all ten EEGNet comparison repeats. Main Table 4 additionally contains its learned-embedding/logistic-head row. Neither changes the original primary result.
