# EEG incremental predictive utility

Computational companion for *Incremental Predictive Utility of EEG Representations
Beyond Behavioral and Object-Gaze Measures in Pediatric ADHD*, by Alptekin Cinbat,
Ümit Şentürk, and İbrahim Yücedağ.

This study asks whether EEG improves prediction beyond information already
available from the same attention task. It compares paired learning procedures in
96 BALLADEER participants, with identical participant-level outer partitions and
training-only preprocessing, tuning, and representation selection. The target is
the recorded study-group field, not an independently verified clinical diagnosis.

## Main findings

The prespecified spectral/covariance procedure worsened held-out log loss:
**ΔLL = −0.0288**, defined as baseline minus EEG-augmented loss. Positive values
favor EEG. Only one of ten dependent repeats was positive; these repeats are not
independent samples.

A later permutation-entropy/Hjorth representation had ΔLL = +0.0267, but the gain
weakened after device and signal-quality adjustment and was similar to that from
Fourier-phase controls (+0.0271 on average). It does not establish a specific
physiological mechanism. The spectral/covariance result remained unfavorable in
the tested SVM and forest procedures; this does not exclude other model or
representation choices. Device-transfer means combined opposing class-specific
changes. An independent adult-data conceptual comparison (50 training, 36 test
participants) showed no consistent EEG benefit. It did not transport the pediatric
model. Controlled simulations distinguish available information from fitted gain.

One subsequent EEGNet-like learned-representation pipeline also worsened held-out prediction: baseline log loss 0.6244, augmented log loss 0.7250, **Delta LL = -0.1006**, with no favorable repeat. Its conditional participant-resampling range was [-0.2368, +0.0267], which includes zero and omits training/selection uncertainty. This fixed-budget result does not establish an information bound for learned EEG representations.

The central contribution is a reproducible matched sensor-addition assessment,
including negative and exploratory findings. No clinical deployment, equivalence,
universal absence of EEG information, or new optimal algorithm is claimed.

## Repository layout

| Path | Contents and status |
| --- | --- |
| `reference_release/` | Preserved computational release v1.1.0: original code, derived features, saved predictions/results and original notices. |
| `extensions/device_transfer/` | Later source-device/target-device stress test, plan, code, complete results and conditional draws. |
| `extensions/utility/` | Later learner, training-size, simulation and adult-data comparisons; all planned outcomes retained. |
| `extensions/eegnet/` | One later compact learned-representation comparison; pre-fit plan, all nested encoders and predictions. |
| `article_assets/` | Current manuscript figures/tables and reporting summaries, with an explicit figure map. |
| `tools/`, `tests/` | Portable entry points and focused checks, writing to separate output directories. |
| `docs/` | Commands, data/provenance, analysis-status and verification boundaries. |
| `FILE_MANIFEST.json` | SHA-256 identities of the distributed files. |

## Installation and reproduction

The main scientific environment uses Python 3.14.3 and the exact versions in
`requirements.txt`. Create a virtual environment and install those dependencies:

```text
python -m venv ../eeg-env
```

Activate the environment (`../eeg-env/Scripts/Activate.ps1` in PowerShell or
`source ../eeg-env/bin/activate` in a POSIX shell), then run
`python -m pip install -r requirements.txt`. From this repository root, use a
**new external directory for each command**:

```text
python -B reproduce.py check --output ../eeg_check
python -B reproduce.py verify-results --output ../eeg_results_check
python -B reproduce.py assets --output ../eeg_display_rebuild
python -B -m unittest discover -s tests -v
```

The [command guide](docs/COMMANDS.md) separates saved-result checks, selected refits
and full/raw-data reproduction. Supplied feature tables suffice for the first
three commands; raw EEG is needed only for the raw pipeline. Existing outputs are
never silently treated as valid caches. Input identity and output-location checks
fail explicitly rather than mixing data or configurations. Historical entry
points are preserved as source records; use the root wrapper for portable runs.

The compact learned-representation comparison has a separate [EEGNet guide](docs/EEGNET.md) and `eegnet.py` entry point. Raw window reconstruction uses the original main environment; neural training uses a separately pinned Python 3.12/PyTorch environment. No pretrained weights or additional cohort are used.

## Data, chronology and limits

BALLADEER is available from [Figshare](https://doi.org/10.6084/m9.figshare.28676042).
Adult source data are from [OSF EWKGV](https://doi.org/10.17605/OSF.IO/EWKGV).
See [data and provenance](docs/DATA_AND_PROVENANCE.md), [analysis status](docs/ANALYSIS_STATUS.md)
and [license notices](NOTICE.md). Raw-export identification and historical archival
limits are stated explicitly; successful saved-feature checks are not raw-data
end-to-end validation.

This companion extends the archived [v1.1.0 release](https://doi.org/10.5281/zenodo.23199244)
with the additional analyses reported in the manuscript, including the EEGNet-like
comparison. The archive DOI identifies v1.1.0; the later analyses are available in
this GitHub repository. Use `CITATION.cff` to cite the current software version and
acknowledge both source studies.

The primary analysis was registered at [OSF 9jp2h](https://osf.io/9jp2h/), currently
embargoed. Later exploratory analyses are documented separately in the
[analysis-status guide](docs/ANALYSIS_STATUS.md).

Original code/tests are MIT; owned documentation, figures, tables and derived
content are CC BY 4.0. Third-party material retains its own terms, detailed in
[NOTICE.md](NOTICE.md). Manuscripts and journal declarations are supplied separately.
