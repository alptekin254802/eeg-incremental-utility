# Targeted age, bandwidth, and prior-odds controls

This addendum evaluates three bounded exploratory questions after the original
study outcomes were known. The plan was recorded before the new controls were
computed. It is not an amendment to the original preregistration.

- Flexible age: replace linear age with the fixed restricted cubic spline in three matched comparisons.
- Bandwidth: apply one additional 30 Hz low-pass filter and refit two entropy/Hjorth comparisons using the original accepted windows.
- Prior odds: apply the same source-to-target prior-log-odds shift to both saved models in four device-transfer comparisons. Target labels make this an oracle diagnostic.

All code, predictions, selected hyperparameters, derived filtered features,
conditional draws and numerical summaries are in [`targeted_controls/`](../targeted_controls/).
Every file from the dated separate addendum is preserved byte-for-byte, including
its original README and manifest. The frozen README refers to the earlier archive
and earlier supplementary table numbering. This guide supplies the current
repository paths and mappings without rewriting that historical record.

## Environment and checks

Use Python 3.14.3 and the pinned packages in
[`targeted_controls/requirements.txt`](../targeted_controls/requirements.txt).
From the repository root, in an activated environment:

```text
python -m pip install -r targeted_controls/requirements.txt
python -B targeted_controls/code/run.py verify --release companion/reference_release --transfer companion/extensions/device_transfer --output ../eeg-targeted-check
```

The output directory must not already exist. This command validates the bundled
file manifest, frozen analysis plan, computational code identities and dependency
hashes. It recomputes the original and oracle device-transfer scores, all 5,000-draw
conditional summaries, saved nested-prediction summaries and tuning selections,
then regenerates both tables. It compares all generated CSV values and tables with
the supplied outputs. This is saved-result numerical verification and conditional
resampling replay, not predictive model refitting or raw-data reconstruction.

## Full recomputation of this analysis family

These commands are separate from quick verification. Use a new output directory:

```text
python -B targeted_controls/code/run.py init --release companion/reference_release --transfer companion/extensions/device_transfer --output ../eeg-targeted-run
python -B targeted_controls/code/run.py age --release companion/reference_release --transfer companion/extensions/device_transfer --output ../eeg-targeted-run --workers 4
python -B targeted_controls/code/run.py extract --release companion/reference_release --transfer companion/extensions/device_transfer --output ../eeg-targeted-run --raw RAW_DATASET_DIR --workers 2
python -B targeted_controls/code/run.py bandwidth --release companion/reference_release --transfer companion/extensions/device_transfer --output ../eeg-targeted-run --workers 4
python -B targeted_controls/code/run.py offset --release companion/reference_release --transfer companion/extensions/device_transfer --output ../eeg-targeted-run
python -B targeted_controls/code/run.py summarize --release companion/reference_release --transfer companion/extensions/device_transfer --output ../eeg-targeted-run
```

Replace `RAW_DATASET_DIR` with the extracted [BALLADEER source dataset](https://doi.org/10.6084/m9.figshare.28676042).
The 96 expected relative raw filenames and checksums are recorded in
[`RAW_RECONSTRUCTION.json`](../targeted_controls/results/extract/RAW_RECONSTRUCTION.json).
Missing or mismatching inputs fail explicitly. No raw EEG is redistributed.
The bandwidth stage requires the matching reconstructed features. All scientific
seeds, folds, cutoffs and model choices remain those of the recorded analysis.

## Results and article mapping

| Files | Purpose | Current supplement | Technical note |
| --- | --- | --- | --- |
| `tables/supp_targeted.tex`, `results/summary/` | All five age/bandwidth comparisons and conditional summaries | Table S11, Section S10 | Targeted sensitivities |
| `tables/supp_offset.tex`, `results/offset/` | All four paired transfer comparisons, before and after the common offset | Table S8, Section S5 | Prior-odds diagnostic |
| `results/age/`, `results/bandwidth/` | Every inner score, outer prediction and tuning/selector decision | Underlying computation | Complete numerical record |
| `results/extract/` | Filtered features and original-feature/window reconstruction evidence | Underlying computation | Preprocessing details |
| `results/ANALYSIS_PLAN.txt`, `results/audit/` | Dated plan, timing, input and implementation identities | Exploratory status | Provenance |

The original table descriptions used S19 and S20 for the targeted and offset
tables respectively; the current journal supplement uses S11 and S8. The numerical
table sources are unchanged. Positive differences favor EEG. Conditional ranges
hold predictions fixed and omit fitting and selection uncertainty. Age and
bandwidth were evaluated separately. The oracle correction cannot be deployed
as assessed and does not isolate a device effect.

## Archival scope and licenses

The technical note is archived in v1.2.2 (DOI 10.5281/zenodo.23929744). The
computational addendum in this checkout was added after that snapshot and is
prepared for the next release, 1.2.3. It is not already published under the
v1.2.2 DOI. Cite the exact commit until the corresponding release is archived.

Original code is MIT licensed (`targeted_controls/LICENSE_CODE.txt`). Original
documentation, tables and derived results are CC BY 4.0; source-data and
third-party terms remain applicable. See the preserved addendum README and
[`companion/NOTICE.md`](../companion/NOTICE.md).
