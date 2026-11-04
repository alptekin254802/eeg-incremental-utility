# Incremental predictive utility of EEG representations

Computational companion for *Incremental Predictive Utility of EEG Representations
Beyond Behavioral and Object-Gaze Measures in Pediatric ADHD*, by Alptekin Cinbat,
Ümit Şentürk, and İbrahim Yücedağ.

Does EEG improve prediction beyond demographics, task behavior, and object-gaze
measures? The study compares matched learning procedures in 96 BALLADEER
participants using repeated participant-level nested cross-validation. The target
is recorded study-group membership, rather than independently verified diagnosis.

## Findings

The prespecified EEG procedure worsened held-out log loss: **ΔLL = −0.0288**,
defined as baseline minus EEG-augmented loss. Positive values favor EEG.
The exploratory permutation-entropy/Hjorth result was favorable (**+0.0267**),
but weakened with acquisition adjustment and was similar in Fourier-phase controls
(**+0.0271**). It does not establish a specific physiological mechanism.

Selection accounted for about three quarters of the primary deficit: the selector
performed worse than each fixed representation. Exploratory diagnosis-status
entropy/Hjorth gains were +0.0094 without adjustment and +0.0110 with device and
quality adjustment. Two cross-fitted single-logit EEGNet-like heads gave small
matched increments (+0.0019 and +0.0053); both conditional ranges include zero.
Phase-lag's observed +0.0127 gain exceeded 19 of 20 phase controls (mean -0.0114),
unlike entropy/Hjorth. These descriptive follow-ups are in
[`exploratory_r2_2026-11-05/`](exploratory_r2_2026-11-05/README.md).
They do not validate a clinical diagnostic system or establish an upper bound on EEG utility.

## Current code and results

The original and earlier exploratory computations are in **[`companion/`](companion/README.md)**. This preserved bundle includes
the preserved v1.1.0 analysis, subsequent exploratory extensions, all outcomes,
and the display set archived with version 1.2.1 (12 figures and 22 table sources).

| Path | Contents |
| --- | --- |
| [`companion/reference_release/`](companion/reference_release/) | Preserved v1.1.0 code, inputs, results, and notices |
| [`companion/extensions/`](companion/extensions/) | Device-transfer, learner, simulation, adult-data, and EEGNet-like analyses |
| [`companion/article_assets/`](companion/article_assets/) | Figures and tables archived with version 1.2.1 |
| [`companion/docs/`](companion/docs/) | Reproduction, data provenance, analysis status, and verification scope |
| [`companion/FILE_MANIFEST.json`](companion/FILE_MANIFEST.json) | File identities for the preserved companion |
| [`targeted_controls/`](docs/TARGETED_CONTROLS.md) | Later age, 30 Hz bandwidth, and common prior-odds controls: complete code, derived inputs, predictions, and results |

The [technical methods and complete numerical results](docs/TECHNICAL_METHODS.md)
collect implementation details, all comparison families, and additional diagnostic
figures relocated from the journal supplement. The note was first archived in
[version 1.2.2](https://doi.org/10.5281/zenodo.23929744). Version 1.2.6 adds the four exploratory follow-ups and their plan/code/output links, while preserving the historical numerical tables and figures as reference.
Its [file manifest](docs/TECHNICAL_METHODS_MANIFEST.json) records the current identities.
Current journal numbering differs from the historical display set.

The later age, bandwidth, and common prior-odds computational addendum is now
included in [`targeted_controls/`](docs/TARGETED_CONTROLS.md). It preserves every
original code, plan, prediction, result, and input-identity file from the separate
addendum without alteration. Unadjusted entropy/Hjorth gains remained positive
under the age and bandwidth sensitivities, but were smaller with device/quality
adjustment. Common prior correction reversed three of four transfer increments.
These exploratory controls do not identify a physiological mechanism.
These computations are archived in [version 1.2.3](https://doi.org/10.5281/zenodo.23930239); earlier archives remain unchanged.

The original files at the repository root remain available for compatibility.
Use the companion directory for the earlier analyses and the targeted-control
guide below for the later controls. The exact archived
layout remains available at the [v1.1.0 tag](https://github.com/alptekin254802/eeg-incremental-utility/tree/v1.1.0).

## Current journal displays

[`journal_assets/`](journal_assets/README.md) reproduces the current 16 tables
and 8 figures, including the evaluation-workflow table and the compact
device-transfer plot. It checks frozen input identities, validates the new cells against round-1 outputs,
and reproduces every current table and figure byte-for-byte. Earlier computations
remain unchanged; version 1.2.6 adds the four dated exploratory follow-ups.

```text
python -B journal_assets/build.py --output ../eeg-journal-assets
```

## Getting started

Use Python 3.14.3 and the pinned dependencies. From a clone of this repository:

```text
cd companion
python -m venv ../../eeg-env
```

Activate `../../eeg-env/Scripts/Activate.ps1` in PowerShell or
`source ../../eeg-env/bin/activate` in a POSIX shell, then run:

```text
python -m pip install -r requirements.txt
python -B reproduce.py check --output ../../eeg-check
python -B reproduce.py verify-results --output ../../eeg-result-check
python -B reproduce.py assets --output ../../eeg-assets
python -B -m unittest discover -s tests -v
```

Use a new output directory for each command. These checks and display generation
use distributed derived inputs and results; they do not train the complete study.
The [command guide](companion/docs/COMMANDS.md) distinguishes saved-result checks,
selected refits, and raw-data reproduction. The [EEGNet guide](companion/docs/EEGNET.md)
describes its separate PyTorch environment, raw-window preparation, and training.
The [verification record](companion/docs/VERIFICATION.md) states what was executed
and what remains outside the checks.

### Later targeted controls

From the repository root, with the same pinned dependencies installed:

```text
python -B targeted_controls/code/run.py verify --release companion/reference_release --transfer companion/extensions/device_transfer --output ../eeg-targeted-check
```

Use a new output directory. This checks file identities, replays conditional
resampling from saved predictions, verifies all comparisons and tuning selections,
and regenerates both tables. It does not refit predictive models or read raw EEG.
The [targeted-control guide](docs/TARGETED_CONTROLS.md) gives the complete refitting
and raw-feature reconstruction commands and maps historical table numbers to the
current article.

## Data and citation

This is version 1.2.6. Earlier analyses remain preserved, including
[version 1.2.5](https://doi.org/10.5281/zenodo.23959046). The four new exploratory
follow-ups retain their pre-fit plans, executed code, frozen inputs, predictions,
tuning records, and complete outcomes. Use the exact version DOI linked from the
[v1.2.6 release](https://github.com/alptekin254802/eeg-incremental-utility/releases/tag/v1.2.6)
or its commit; the [concept DOI](https://doi.org/10.5281/zenodo.23146468) identifies
the series. See [CITATION.md](CITATION.md).

Source data are available from [BALLADEER on Figshare](https://doi.org/10.6084/m9.figshare.28676042)
and [the adult study on OSF](https://doi.org/10.17605/OSF.IO/EWKGV).
Please cite the relevant source studies when using their data. The primary analysis
was registered at [OSF 9jp2h](https://osf.io/9jp2h/), currently embargoed; later
extensions are exploratory. See [data and provenance](companion/docs/DATA_AND_PROVENANCE.md)
and [analysis status](companion/docs/ANALYSIS_STATUS.md).

Original code and tests are MIT licensed; original companion documentation,
figures, tables, and derived content are CC BY 4.0. Third-party material retains
its own terms in the [license notices](companion/NOTICE.md). Manuscripts and
journal declarations are distributed separately from this software repository.
