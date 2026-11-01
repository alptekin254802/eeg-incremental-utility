# Reproduction commands

Run these commands from the complete package root with Python 3.14.3 and the pinned dependencies in `requirements.txt`. Every command requires the package's `FILE_MANIFEST.json`. The wrapper verifies all listed files before and after execution and rejects missing, altered, or additional files. Use `python -B` to avoid creating bytecode files in the versioned source tree.

Each `--output` must name a **new or empty directory outside both the package and any raw dataset**. An ancestor of either input is also prohibited. Completed or partial output directories are never resumed or overwritten. Preserve an interrupted output for inspection and choose a new directory after correcting the cause. A changed package, code file, plan, input, or dependency is not silently accepted as a cached run.

The wrapper records the package manifest identity, software versions, command configuration, and a code/data fingerprint in `CONFIG.json`. The fingerprint excludes root documentation, allowing a later documentation-only assembly to be compared with a previously validated computation. It is an identity aid, not permission to reuse an existing output directory. `SOURCE_UNCHANGED.json` verifies the original file and directory inventory after execution. A successful command also writes `COMPLETED.json`.

## Checks and saved-result verification

```text
python -B reproduce.py check --output ../check-output
python -B reproduce.py verify-results --output ../saved-result-check
```

`check` verifies package and nested manifests, plans, downloaded adult files, and the original release's frozen measurement structure. File identity is distinct from numerical reproduction.

`verify-results` re-scores primary, fixed spectral, E3/E4/E5, post-hoc, phase-control, empirical-extension and simulation probability records, checks split nesting and selector choices, and replays the recorded device/adult conditional resampling. It also invokes the original fixed-prediction resampling and numerical scalar-calibration follow-ups. The arithmetic tolerance is `1e-10`; maximum discrepancies are retained. The scalar calibration diagnostics are fitted to saved probabilities. **No predictive classifier is fitted and no raw EEG is processed by this command.** The historical diagnosis-sensitivity individual predictions and selected regularization values are absent from the original archive and therefore cannot receive the same historical prediction-identity check.

## Current article assets

```text
python -B reproduce.py assets --output ../article-asset-rebuild
```

This command rebuilds the six original figures, device-transfer figure, four utility-extension figures, the EEGNet comparison, and their corresponding tables from saved results. The original display builder verifies its full numerical/label record. The new wrapper writes `assets/ASSET_MAP.json` for the current article:

| Current figure | File under `assets/figures/` |
|---|---|
| 1 | `repeat_increments.pdf` |
| 2 | `acquisition_phase_controls.pdf` |
| 3 | `component_increments.pdf` |
| 4 | `device_transfer.pdf` |
| 5 | `independent_adult_increment.pdf` |
| S1 | `calibration.pdf` |
| S2 | `participant_changes.pdf` |
| S3 | `coefficients.pdf` |
| S4 | `model_family_sensitivity.pdf` |
| S5 | `learning_curves.pdf` |
| S6 | `controlled_simulations.pdf` |

The preserved old scripts' console wording may use historical figure positions; their output is retained in `logs/` and does not control this map. The wrapper does not renumber archived captions or alter manuscript files. If supplied, the separate retained-duration tool also rebuilds `assets/quality_summary/`, and `tools/build_reporting_assets.py` rebuilds the later descriptive tables under `assets/reporting/`. These additions do not change the eleven figure identities.

One historical footnote in the original comparison table points to Section S9. After the preserved builder and its numerical checks finish, the new wrapper changes that pointer in the output copy to `\ref{sec:s-data}`, matching the current article. `ASSET_MAP.json` records this adaptation; numerical cells, captions, and the preserved generator are unchanged.

PDF metadata and rendering-library serialization can change PDF hashes. Compare numerical records and rendered figures, not PDF bytes alone. Asset regeneration reads the saved results; it does not render results from a new model-replay run.

## Saved-probability reporting follow-up

```text
python -B reporting/probability_diagnostics/saved_prediction_diagnostics.py --release-root reference_release --transfer-root extensions/device_transfer --extension-root extensions/utility --output ../probability-diagnostics
```

Keep the accompanying `ANALYSIS_PLAN.txt` beside this script. This separate post-hoc command recalculates probability/tail summaries and a bounded selection of recorded conditional bootstrap draws. It fits no predictive model, retains every planned threshold/comparison, and does not independently recreate the full bootstrap distributions. The full conditional-distribution replay described above belongs to `verify-results`; the two verification scopes must not be conflated. Use a fresh external output directory for this standalone command as well.

## Selected model replays

```text
python -B reproduce.py replay-empirical --first-only --workers 1 --output ../empirical-first-task
python -B reproduce.py replay-simulation --first-only --workers 1 --output ../simulation-first-task
```

`--first-only` runs one deterministic task selected by the preserved original task ordering. The empirical task is repeat 0, fold 0, full training size, unadjusted; the simulation task is redundant features, replicate 0, training size 96. These are diagnostic checks, not a full reproduction. New tuning, split, probability and summary records are compared against the saved task with the original absolute `1e-7` numerical tolerance; categorical choices and split membership must agree. Simulation comparison uses every array retained in the compact source archive. It does not require omitted duplicate synthetic covariates.

## Complete frozen-feature refits

```text
python -B reproduce.py replay-transfer --output ../transfer-refit
python -B reproduce.py replay-empirical --workers 4 --output ../empirical-refit
python -B reproduce.py replay-simulation --workers 4 --output ../simulation-refit
python -B reproduce.py replay-adult --output ../adult-refit
```

These commands run new predictive fits from distributed feature inputs. They do not re-extract pediatric EEG. Transfer invokes the preserved 260-fit analysis and its deterministic CSV/arithmetic verifier. The empirical and simulation commands without `--first-only` run the complete 200-task and 240-task programs. The adult command repeats all six preplanned paired comparisons, tuning, and conditional test resampling. Published plans, seeds, model grids, cohort exclusions and reference results remain unchanged. A command being available here does not imply that a complete rerun was executed during package preparation; executed validation is reported separately.

## Raw-data pipeline

```text
python -B reproduce.py raw-pipeline --dataset /path/to/BALLADEER/dataset --workers 2 --output ../fresh-raw-reproduction
```

`--dataset` is mandatory and must identify the directory containing `users_demographics.json` and the participant recording folders. The wrapper does not download data. The original raw pipeline rebuilds measurement eligibility, features, outcome linkage, splits and models, with its own fresh-input and checkpoint contracts. The wrapper limits this route to at most two workers. The raw dataset and output must be disjoint from each other and from the package. Only this command can initiate the complete raw-extraction-to-model route; saved-feature replays and figure regeneration cannot certify it.

## Read-only source handling and tests

Legacy scripts create runtime folders or write verification reports beside their source. Commands that invoke those scripts first copy the entire package to `<output>/workspace`, then normalize copied manifest path separators and local release locations. Thus all legacy side effects stay inside the external output. Original sources, plans, seeds, results and their manifests are never rewritten. `WORKSPACE_ROUTING.json` lists the limited copied-metadata routing changes. The copied manifests continue to describe original bytes and are not presented as new immutable input manifests after routing changes.

```text
python -B -m unittest discover -s tests -p "test_portable_pipeline.py" -v
```

The tests cover path traversal, both stored separator conventions, manifest mismatches, output/input overlap, stale output/configuration rejection, source-preserving staging, the computation fingerprint, and current figure identities. They do not rerun the scientific model campaign. Native POSIX path execution should be reported separately from the unit test that verifies slash normalization; neither implies the other.

## Learned EEG representation

See [EEGNET.md](EEGNET.md) for the separate, fully nested 300-encoder comparison. Its `verify` command checks the complete saved predictions and conditional draws without training. Its `prepare`, `train`, and selected `replay` commands have different scopes and require the stated two environments.
