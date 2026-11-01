# Result files: performance comparisons and diagnostics

All paths below are relative to the repository root. Figure and table commands read these result files directly.

The generated `<output>/display_assets/tables/identifier_crosswalk.tex` (Supplementary Table S2) maps computational identifiers to method names. The result files keep their original column names and formats.

The main article presents the cohort/device description in Table 1, the non-EEG and EEG performance comparisons in Table 2, and six matched acquisition-adjustment contrasts in Table 3. Supplementary Tables S3 and S6 give the full performance results and contrasts. The component comparisons are shown in Main Figure 3; calibration, participant-level loss changes, and coefficients are Supplementary Figures S1–S3. Supplementary Table S1 retains the participant flow.

## Cohort records

The cohort tables use `audit/preregistered_analysis/OUTCOME_LINKAGE_MANIFEST.csv`, the participant roster and preprocessing QC in `audit/stage1_blind_feature_extraction/`, and `audit/cohort_screen/robots_final_feasibility.csv`. They summarize the recorded groups, demographics, devices and eligibility decisions without fitting models.

## Primary and representation results

- `audit/preregistered_analysis/PRIMARY_REPEAT_SUMMARY.csv`: primary baseline and EEG procedure, one row per repeat.
- `audit/preregistered_analysis/PREREGISTERED_RESULTS_MASTER_TABLE.csv`: summary of primary and secondary results; status fields identify each representation's role in the analysis.
- `audit/preregistered_analysis/MODALITY_LADDER_CONSOLIDATION.csv`: demographic, behavior, and object-gaze model performance and sequential differences.
- `audit/preregistered_analysis/E0_CORE_SENSITIVITY.csv`, `audit/preregistered_analysis/FIXED_E1_SECONDARY.csv`, `audit/preregistered_analysis/FIXED_E2_SECONDARY.csv`: full-precision repeat results for the prespecified fixed representations.
- `audit/preregistered_analysis/EXPLORATORY_MODALITY_LADDER.csv`: full-precision repeat results for the non-EEG models. Figures and tables use means calculated from these files and the fixed-representation files. The consolidation tables provide the original summaries and analysis-status labels.
- `audit/preregistered_analysis/DIAGNOSED_SENSITIVITY.csv`: the separate yes/no diagnosis-status outcome in 84 participants, with newly stratified folds.
- `audit/preregistered_analysis/CALIBRATION_RESULTS.csv`: original scalar estimates/failures and four-bin reliability summaries. Failure flags concern the additional scalar calibration regressions.
- `audit/exploratory_eeg_battery/REPEAT_LEVEL_RESULTS.csv` and `results/e5/REPEAT_RESULTS.csv`: phase-lag, entropy/Hjorth, and band-grouped spectral-state results. The display code concatenates their rows in memory.
- `expected/posthoc/results/participant_loss_diagnostics.csv`: saved participant mean-loss changes. The display code selects the primary and entropy/Hjorth models and sorts participants independently within each model in memory. The plotted records contain only model, rank, and loss change; no participant identifiers or clinical/acquisition metadata are exported.

See `display_assets/figure_source/REPRODUCIBILITY.md` for analysis settings and figure/table commands.

The component, acquisition, EEG-optional-selection, and phase-control comparisons were specified after the primary and additional-representation results were known.
These analyses are exploratory. All ten repetitions of nested cross-validation
use the same 96 participants, so rows from different repeats are not independent.

## Files

- `expected/posthoc/results/comparison_summary.csv`: all 35 model comparisons, including archived reference results.
- `expected/posthoc/results/repeat_results.csv`: the corresponding repeat-specific performance and differences.
- `expected/posthoc/signal_controls/comparison_summary.csv`: all 40 contrasts from 20 Fourier-phase control datasets, against M2 and observed E4.
- `expected/posthoc/signal_controls/repeat_results.csv`: all corresponding repeat-specific results.
- `expected/posthoc/results/coefficient_summary.csv`, `expected/posthoc/results/outer_coefficients.csv`: standardized coefficients and descriptive summaries of 50 outer fits for M2 and M2+E4. Ranges and sign frequencies are not inferential intervals.
- `expected/posthoc/results/e4_spearman.csv`: Spearman correlations among the nine observed E4 features.
- `expected/posthoc/results/device_by_group.csv`: hardware distribution by study group.
- `expected/posthoc/results/device_group_loss_diagnostics.csv`: participant-repeat-averaged held-out losses summarized within device and study group, without subgroup-specific fitting.
- `expected/posthoc/results/device_group_quality_distributions.csv`: descriptive recording-quality distributions. This file has a two-row column header followed by a row naming the device/group index. Read it with pandas `header=[0,1], index_col=[0,1]`.
- `expected/posthoc/results/e4_device_quality_correlations.csv`: descriptive feature associations with device and quality variables.
- `expected/posthoc/results/participant_diagnostic_summary.csv`: cohort-level summaries of participant-average gains/losses, including tail contributions; no participant exclusions were based on these results.
- `expected/posthoc/results/logit_term_summary.csv`: distribution across participants of repeat-averaged absolute logit contributions, including changes in refitted baseline terms. These are not signed mean effects or additive shares of predictive improvement.
- `expected/posthoc/signal_controls/surrogate_feature_diagnostics.csv`: observed versus surrogate feature summaries.

## Conventions

`baseline` and `augmented` specify the actual comparison direction. Positive
`delta_ll` and `delta_brier` indicate lower loss for the evaluated model;
positive `delta_auroc` indicates higher AUROC. Some direct contrasts use full
E4 as baseline. In signal comparisons `surrogate_XX_vs_observed_E4` uses the
control as baseline, so positive values favor observed E4. `positive_repeats`
counts favorable log-loss differences among the ten dependent repeats.

EEG model identifiers implicitly include the seven M2 predictors. `D` adds a
device indicator to both models. `DQ` additionally adds clean-window fraction
and interpolated-channel count. `original_*` denotes archived reference results.
`only_*` and `without_*` indicate separately tuned E4 component models.
`relative9` uses nine regional relative-power variables; `PCA9` uses nine
training-derived E0 components. `optout` selects M2 or M2+E0/E1/E2 within training.
Study-group coding is 0 = Control, 1 = ADHD/Experimental; it is not the separate
diagnosis-status field. Surrogate dataset indices run from 0 through 19.

## Figure/table generation and file checks

`display_assets/figure_source/ASSET_VALUES.json` records input paths and hashes, plotted values, table cells and labels. `RELEASE_MANIFEST.json` provides checksums for the distributed files. Values are rounded only for presentation; differences are calculated from the full-precision results.

## Spectral-state features and results

- `results/e5/E5_BAND_ORDERED_FEATURES.csv`: theta, alpha, and beta feature blocks, each comprising three regions and three summaries.
- `results/e5/INNER_SELECTION.csv`: tuning choices within the training partitions.
- `results/e5/OUTER_PREDICTIONS.csv`: held-out predictions from the outer folds.
- `results/e5/REPEAT_RESULTS.csv`: performance estimates and paired differences for each repeat.
- `results/e5/RESULTS.json`: mean performance, selected regularization values, and computational fit counts.

This representation was evaluated post hoc using band-specific PCA fitted within training data. Repeat indices are zero-based in the files and one-based in figures and tables. Computational identifiers are mapped in the generated `<output>/display_assets/tables/identifier_crosswalk.tex` (Supplementary Table S2).

## Numerical follow-up

- `results/numerical_followups/CALIBRATION_NUMERICAL.csv`: damped-Newton solutions of the original unpenalized calibration model, alternative-solver and gradient checks, and original failure flags. No predictive model was refitted.
- `results/numerical_followups/FIXED_OOF_RANGES.csv` and `results/numerical_followups/FIXED_OOF_SPECIFICATION.json`: post-hoc resampling of paired participant mean losses with all fits and predictions held fixed. The ranges describe variation conditional on those fits; they are not confirmatory confidence intervals and have no associated p-values.

From the repository root, `python reproduce.py calibration-numerical --output ../reproduction` and `python reproduce.py descriptive-resampling --output ../reproduction` regenerate these follow-ups. The original result CSV files are unchanged.
