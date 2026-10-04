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
| Figures | `figures --output ../reproduction` | `figures/`: Figures 1–6 and S1–S5; PDF/SVG or PDF/PNG |
| Tables | `tables --output ../reproduction` | `tables/`: numerical source tables, Table 2 and S7/S8 LaTeX |
| Saved-input reproduction sequence | `all --dataset ../BALLADEER/dataset --output ../reproduction --jobs 2` | All the above in dependency order; fits consume saved features |
| Fresh raw-to-model pipeline | `raw-pipeline --dataset ../BALLADEER/dataset --output ../fresh-reproduction --jobs 2` | New screen, features, folds, primary/diagnosis/E3–E5/post-hoc/control fits; input manifests and access records |

Start a new output directory for each full reproduction. E5 refuses to overwrite a completed run. Other stage commands only write inside the chosen output tree. `summarize`, `figures` and `tables` can work directly from the released reference outputs; they record this starting point if no regenerated posthoc results exist. These commands alone do not refit any model.

## What is and is not regenerated

The raw screen reads the dataset and verifies the exact 126→98 roster. The feature command independently re-extracts the EEG and non-EEG predictors and checks them against saved artifacts; it starts at the archived 98-person roster, which the screen command verifies. Model commands deliberately consume the immutable feature checkpoints protected by input identity checks. Thus `all` validates the raw-to-checkpoint link and refits from the checkpoints; it is not a single pipeline passing newly serialized feature files directly into the learner. Input identity checks and numerical agreement checks are separate.

Figure and table commands consume released numerical summaries (including E5), or the newly computed posthoc summaries where present. Table S1 describes cohort accounting; Tables S2 and S3 use the primary model tables; the complete full-precision source tables remain in `audit/preregistered_analysis`. Exact manuscript text and LaTeX layout are maintained separately from this computational companion.

## Numerical comparisons

Source and result identities are recorded in `RELEASE_MANIFEST.json`. Numerical reproduction is assessed separately using the tolerances above; a different PDF hash can reflect metadata or font serialization. Compare plotted values, labels and axes when checking regenerated figures. See [known reproduction limits](REPRODUCTION_LIMITS.md) before interpreting an intermediate-score or calibration mismatch.

## Recalculate reference summary tables

The following commands recompute the primary master table, modality ladder and E3–E5 summary from the supplied repeat-level results.

```text
python -B audit/preregistered_analysis/consolidate_results.py --output ../reproduction/reference_tables
python -B audit/exploratory_eeg_battery/create_master_table.py --output ../reproduction/reference_tables
```

Hashes embedded in result-provenance records identify the calculation that produced those results. `RELEASE_MANIFEST.json` identifies the distributed files.
