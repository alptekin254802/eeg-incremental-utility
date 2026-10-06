# Reproduction guide

Use `reproduce.py`, with `--output` outside the release. Paths are resolved from the released package and the dataset/output arguments. The `audit/` directory contains reference inputs, not writable destinations. The measurement and estimator identity checks remain active in saved-input replay commands. The `raw-pipeline` command connects fresh extraction to fresh model fits; see [the full input-chain contract](FRESH_RAW_REPLAY.md).

## Commands and boundaries

Replace the example dataset/output paths with your local paths. Python 3.14.3 and the pinned scientific packages were used; `--jobs 2` controls process parallelism. Numerical results should agree within `1e-7` for probabilities/model losses and `atol=1e-10, rtol=1e-8` for re-extracted features; selected models must agree exactly.

| Stage | Command after `python reproduce.py` | Main outputs |
|---|---|---|
| Identity checks | `check --output ../reproduction` | Release and measurement hashes, 96-person matrix check |
| Raw eligibility | `raw-screen --dataset ../BALLADEER/dataset --output ../reproduction` | `raw_screen/`: parse all 126 complete sessions, reconstruct the 98-person roster, source hashes |
| Raw features | `raw-features --dataset ../BALLADEER/dataset --output ../reproduction` | `raw_features/`: all 98 preprocessing decisions and 96 accepted E0–E5/behavior/gaze reconstructions |
| Primary/secondary models | `models --dataset ../BALLADEER/dataset --output ../reproduction` | `models/`: 450 outer fits, all primary/ladder/diagnosis probabilities, selection and comparison checks |
| Connectivity and entropy/Hjorth (E3/E4) | `exploratory --output ../reproduction` | `exploratory/`: 100 outer fits, tuning records, reference-result checks |
| Spectral-state representation (E5) | `e5 --output ../reproduction` | `e5/`: band-grouped input, 50 outer fits, predictions, tuning and metrics |
| Explanatory models | `posthoc --output ../reproduction` | `posthoc/results/`: 20 fitted models, derived adaptive/opt-out comparisons, 35 summaries and model decompositions |
| Phase controls | `signal-controls --dataset ../BALLADEER/dataset --output ../reproduction` | Requires posthoc first; `posthoc/signal_controls/`: raw reconstruction, 20 phase-control feature sets, 1,000 outer fits and 40 contrasts |
| Diagnostic summaries | `summarize --output ../reproduction` | Participant/logit/device-quality summaries and signal checks |
| Calibration | `calibration --output ../reproduction` | Scalar calibration and four-bin reliability from saved primary outer predictions; failure flags are retained |
| Figures | `figures --output ../reproduction` | `display_assets/figures/`: main Figures 1–3 and supplementary Figures S1–S3, PDF/SVG, with checks of displayed values |
| Tables | `tables --output ../reproduction` | `display_assets/tables/`: main Tables 1–3 and supplementary Tables S1–S8; generated numerical tables and supplied descriptive tables |
| Saved-input reproduction sequence | `all --dataset ../BALLADEER/dataset --output ../reproduction --jobs 2` | Raw-data checks, models, controls, summaries, figures and tables in dependency order; models use saved features; run calibration and resampling follow-ups separately |
| Fresh raw-to-model pipeline | `raw-pipeline --dataset ../BALLADEER/dataset --output ../fresh-reproduction --jobs 2` | New screen, features, folds, primary/diagnosis/E3–E5/post-hoc/control fits; input manifests and access records |

Use a new output directory for each full reproduction. E5 stops if that directory already contains a completed E5 run; other stages write only within the chosen output directory. `summarize` can use the supplied reference results and records when it does so. `figures` and `tables` always read the saved results in `audit/`, `expected/posthoc/` and `results/`, then write to `<output>/display_assets/`. They do not use or overwrite results from a new analysis run. Summary, figure and table commands do not fit models.

## How the commands use data

The raw screen reads all 126 complete sessions and checks that it recovers the original 98-person roster. Starting from that roster, the feature command re-extracts the EEG and non-EEG predictors and compares them with the saved features. Model commands then fit from the saved features, whose hashes are checked before use. Thus, `all` checks the features against raw data and refits the models from the saved files. To fit models directly from newly extracted features, use `raw-pipeline`. File-integrity checks and numerical comparisons are reported separately.

The figure and table code is in `display_assets/`; the [result-file guide](RESULT_FILES.md) lists its inputs. The code combines representation results and ranks participants in memory, without copying the input files. It checks input hashes, plotted values, table cells, labels and figure/table numbering against `display_assets/figure_source/ASSET_VALUES.json`. The `figures` and `tables` commands use this builder; the original generation scripts remain as a record of earlier versions.

## Numerical comparisons

`RELEASE_MANIFEST.json` records checksums for the source and result files. Numerical reproduction is assessed separately using the tolerances above; a different PDF hash can reflect metadata or font serialization. Compare plotted values, labels and axes when checking regenerated figures. See [known reproduction limits](REPRODUCTION_LIMITS.md) before interpreting an intermediate-score or calibration mismatch.

## Recalculate reference summary tables

The following commands recompute the primary master table, modality ladder and E3–E5 summary from the supplied repeat-level results.

```text
python -B audit/preregistered_analysis/consolidate_results.py --output ../reproduction/reference_tables
python -B audit/exploratory_eeg_battery/create_master_table.py --output ../reproduction/reference_tables
```

Hashes in the result records identify the inputs and code used for each calculation. `RELEASE_MANIFEST.json` lists checksums for the files distributed here.


## Numerical follow-ups

```text
python -B reproduce.py calibration-numerical --output ../reproduction
python -B reproduce.py descriptive-resampling --output ../reproduction
```

Both commands use saved participant-level results without refitting predictive classifiers. The `calibration` command still reproduces the original calibration method. Supplementary Sections S5 and S8 describe the follow-up methods and their limitations.

For model checks, add `--strict-intermediates` to `models` or `all` to stop whenever an intermediate score exceeds the comparison tolerance. By default, the command may finish with a warning when differences are limited to the documented unselected `C=100` scores. In that case, `passed` remains `false` in the JSON report because the full numerical check has not passed.

## Figure and table files

See [display_assets/README.md](../display_assets/README.md) for file locations and article numbering. These commands generate figures and tables; the full article and supplement are maintained separately.
