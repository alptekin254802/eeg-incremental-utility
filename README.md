# Incremental Predictive Utility of EEG Representations Beyond Behavioral and Object-Gaze Measures in Pediatric ADHD

Does EEG improve prediction beyond demographics, task behavior and object-gaze measures? This repository contains the analysis code and derived results for our study of 96 BALLADEER participants using repeated nested cross-validation.

The preregistered EEG procedure provided no average improvement over the non-EEG baseline. Post-hoc analyses examined additional EEG representations and the sensitivity of their results to acquisition conditions and signal controls. These exploratory findings require independent validation.

## Getting started

Use Python 3.14.3 and the pinned dependencies:

```text
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` in a POSIX shell, then run:

```text
python -m pip install -r requirements.txt
python -B reproduce.py check --output ../reproduction
```

The check verifies the distributed files, measurement inputs and 96-participant feature matrix. To regenerate the article's figures and numerical tables from the supplied results:

```text
python -B reproduce.py figures --output ../reproduction
python -B reproduce.py tables --output ../reproduction
```

These commands generate main Figures 1–3, supplementary Figures S1–S3 and the numerical tables from the supplied results, without fitting models. Files are written to `<output>/display_assets/`. See the [reproduction guide](docs/REPRODUCIBILITY.md) for individual analyses, expected outputs and comparison tolerances.

## Reproduce from raw recordings

Download BALLADEER from its [Figshare record](https://doi.org/10.6084/m9.figshare.28676042) and arrange it as described in the [data guide](docs/DATA_LAYOUT.md). Replace the example path with your dataset directory:

```text
python -B reproduce.py raw-pipeline --dataset "../BALLADEER/dataset" --output ../fresh-reproduction --jobs 2
```

This route extracts new features and fits the models using those features. Outputs must be outside the repository and raw-data directory. The [raw-data guide](docs/FRESH_RAW_REPLAY.md) explains checkpoints and input verification; [reproduction limits](docs/REPRODUCTION_LIMITS.md) describes known numerical differences.

## Repository contents

| Path | Contents |
|---|---|
| `reproduce.py` and `reproduce/` | Commands for extraction, model fitting, figures, tables and validation |
| `audit/` | Estimator modules, measurement inputs, primary results and exploratory connectivity/entropy results |
| `results/e5/` | Post-hoc spectral-state predictors, held-out predictions, tuning records and results |
| `expected/posthoc/` | Explanatory comparisons and Fourier-phase controls |
| `display_assets/` | Figures, tables and code to regenerate them from saved results |
| `docs/RESULT_FILES.md` | Guide to the result files and their columns |
| `results/numerical_followups/` | Calibration checks and resampling results with predictions held fixed |
| `RELEASE_MANIFEST.json` | SHA-256 checksums for the files in this repository |

The full article and supplement are not distributed in this code repository. The preregistered procedure and the subsequent exploratory analyses are distinguished throughout the documentation.

## Registration and citation

The primary analysis was registered at [OSF 9jp2h](https://osf.io/9jp2h/). The additional EEG representations and explanatory analyses are post hoc.

To cite the software across all published versions, use the Zenodo concept DOI [10.5281/zenodo.23146468](https://doi.org/10.5281/zenodo.23146468). For citation metadata, see [CITATION.cff](CITATION.cff) or [CITATION.md](CITATION.md). Please also cite the [BALLADEER dataset](https://doi.org/10.6084/m9.figshare.28676042) when using its data. Analysis code is available under the [MIT License](LICENSE); [DATA_LICENSE.md](DATA_LICENSE.md) describes the source-data licence and attribution.

## Validation

```text
python -B -m unittest discover -s reproduce -p test_fresh_contract.py -v
```

These tests check separation of saved reference results from fresh model fitting and rejection of altered inputs or incompatible checkpoints. They do not rerun the complete study.

See the [update notes](docs/RELEASE_CANDIDATE.md) for changes to result checks, calibration and resampling.
