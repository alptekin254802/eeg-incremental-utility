Saved-probability diagnostics and selected bootstrap replay

Status
This is a post-hoc, non-blind descriptive supplement. ANALYSIS_PLAN.txt records
the known transfer results, fixed probability-tail rule, comparison family and
verification bounds before the new diagnostics were produced. Existing
predictions remain unchanged. No predictive model, calibration model or raw
measurement extraction is run.

Requirements and command
Python >=3.10, NumPy and pandas. The recorded execution used Python 3.14.3,
NumPy 2.3.4 and pandas 2.3.3. Use the extracted original computational release
and the two reproducibility addenda. Run from this directory:

python -B saved_prediction_diagnostics.py --release-root RELEASE --transfer-root TRANSFER_ADDENDUM --extension-root EXTENSION_ADDENDUM --output NEW_OUTPUT

Replace the capitalized placeholders with local directories. NEW_OUTPUT must
be empty or absent and outside all three source roots. Keep ANALYSIS_PLAN.txt
next to the script. Input paths and hashes are recorded relative to named roots;
no machine-specific source paths are written into the result manifest.
The implementation is independent of the original fitting code: it reads saved
CSV predictions and draws, and computes AUROC by positive-negative rank pairs.

Outputs
PEDIATRIC_PROBABILITY_BY_REPEAT.csv: 60 rows = three designs x two models x ten
repeats. Each row describes 96 probabilities. Includes prevalence, mean and
quantiles of p, log loss, Brier, AUROC and incorrect-tail counts by observed class.

PEDIATRIC_PAIR_BY_REPEAT.csv: 30 rows = three matched comparisons x ten repeats.
Includes paired score increments and the exact contribution of baseline-only,
augmented-only, shared and neither-model incorrect-tail rows to mean delta LL.

PEDIATRIC_PROBABILITY_SUMMARY.csv and PEDIATRIC_PAIR_SUMMARY.csv: six model and
three paired summaries. Scores are averaged after within-repeat evaluation;
AUROC is not calculated from pooled predictions. Quantiles and counts describe
the pooled 960 dependent participant-repeat probabilities, not averaged
participant probabilities or 960 independent people. Unique-participant tail
counts are supplied separately. Positive loss/Brier increments are baseline
minus augmented; positive AUROC increments are augmented minus baseline.

TRANSFER_PROBABILITY_SUMMARY.csv and TRANSFER_PAIR_SUMMARY.csv: both focal EEG
procedures in both transfer directions. Includes source/target composition,
mean probabilities, probability quantiles, absolute metrics, group-specific and
equal-class mean paired changes, and the same tail accounting.

TRANSFER_SELECTED_BOOTSTRAP_REPLAY.csv and ADULT_SELECTED_BOOTSTRAP_REPLAY.csv:
recomputed and saved values plus absolute differences at five selected indices
per comparison. There are 20 transfer comparison/draw rows and 30 adult rows.
Transfer seed is 20261103, draws 0,1,17,999,4999; adult seed is 20261137, draws
0,1,17,999,9999. Original vectorized index matrices are reconstructed to preserve
the exact random stream, but metrics are evaluated only for selected draws.
SELECTED_SAMPLING_INDICES.json records those indices. Transfer indices reference
target rows sorted by participant identifier; adult indices reference published
test-row order. Original saved draw files store metrics, not indices.

VERIFICATION.json: executed checks and explicit omissions. INPUT_MANIFEST.json:
source-relative input hashes, code/plan hashes and software versions. FINDINGS.txt:
concise numeric interpretation. SUGGESTED_SECTION.tex: an integration-ready prose
and table suggestion requiring booktabs and normal manuscript float handling;
it is not a standalone LaTeX document. Its placement and final typesetting should
be checked when integrated into the manuscript.

Recorded verification
All 30 pediatric pair-repeat metric rows agree with original saved results;
maximum absolute error 3.47e-16. Selected transfer and adult draws agree within
1.91e-16 and 1.45e-16, respectively. Inputs retained their pre-read hashes.
Two fresh output directories produced all 13 files byte-for-byte identically;
FRESH_OUTPUT_VERIFICATION.json records that comparison. Existing scientific
inputs and earlier reports were not modified.

Interpretation bounds
Incorrect-tail probabilities use strict p<0.1 for y=1 or p>0.9 for y=0.
This rule is descriptive and is not a diagnostic decision threshold. Tail
contributions divide each subset's summed paired losses by the full assessment
denominator, so the mutually exclusive components sum to the overall mean.
They are not subgroup treatment effects, causal components or an exclusion
analysis. Mean probability close to prevalence does not establish calibration.
Quality-adjusted baselines themselves require EEG acquisition. No clinical
fairness, target recalibration, inferential significance or physiological
specificity is claimed. Full bootstrap metric distributions, fixed-OOF endpoints,
raw signal construction and training-population uncertainty are not independently
recomputed in this bounded follow-up.
