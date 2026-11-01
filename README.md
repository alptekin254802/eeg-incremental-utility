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

Further analyses examined nonlinear learners, device transfer, training size,
controlled simulations, and an independent adult-data conceptual comparison.
One fixed-budget EEGNet-like learned representation also worsened log loss
(**ΔLL = −0.1006**). Its conditional participant-resampling range
**[−0.2368, +0.0267]** includes zero and omits training and selection uncertainty.
These results do not establish an upper bound on EEG utility or validate a clinical
diagnostic system.

## Current code and results

The complete companion is in **[`companion/`](companion/README.md)**. It includes
the preserved v1.1.0 analysis, subsequent exploratory extensions, all outcomes,
and the current article's 12 figures and 22 table sources.

| Path | Contents |
| --- | --- |
| [`companion/reference_release/`](companion/reference_release/) | Preserved v1.1.0 code, inputs, results, and notices |
| [`companion/extensions/`](companion/extensions/) | Device-transfer, learner, simulation, adult-data, and EEGNet-like analyses |
| [`companion/article_assets/`](companion/article_assets/) | Current figures and tables |
| [`companion/docs/`](companion/docs/) | Reproduction, data provenance, analysis status, and verification scope |
| [`companion/FILE_MANIFEST.json`](companion/FILE_MANIFEST.json) | File identities for the complete companion |

The original files at the repository root remain available for compatibility.
Use the companion directory for the current documented workflow. The exact archived
layout remains available at the [v1.1.0 tag](https://github.com/alptekin254802/eeg-incremental-utility/tree/v1.1.0).

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

## Data and citation

This repository extends the archived [v1.1.0 release](https://doi.org/10.5281/zenodo.23199244)
with the additional analyses reported in the manuscript. The archive DOI identifies
v1.1.0; the later analyses are available here on GitHub. Cite the current software
using [CITATION.cff](CITATION.cff), with the version or commit used. See
[CITATION.md](CITATION.md) for the distinction between the current software and the
archived release.

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
