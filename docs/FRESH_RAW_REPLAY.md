# Fresh raw-to-model verification

Create a new Python 3.14.3 virtual environment and install `requirements.txt`
without copying an existing environment or package cache. From this release:

```text
python -m pip install --no-cache-dir -r requirements.txt
python reproduce.py raw-pipeline --dataset /path/to/dataset --output /path/to/new-run --jobs 2
```

The dataset directory follows `docs/DATA_LAYOUT.md`. Outputs must be outside
the release and raw dataset. One or two worker processes are allowed; BLAS
threads are limited to one. No raw downloads are performed.

This command rebuilds the 126-session screen, 98-person measurement roster,
96-person features, outcome linkage, nested folds, 450 primary/diagnosis outer
fits, 150 E3/E4/E5 fits, the 20 post-hoc variants, and all 20 phase
controls. All E3–E5 analyses are post hoc. It uses the released estimator functions and binds their input paths to the
new `fresh_data/` directory. Post-hoc opt-out and reference comparisons use
the **new primary fits** from that directory.

Learning processes reject reads of non-Python files inside the release's
`audit/`, `expected/`, and `results/` directories. The parent wrapper may
hash those files to verify release integrity; their values are not passed to
the learning processes. `access/` records opened paths and per-worker hashes
of actual model input arrays. `FEATURE_MANIFEST.json` connects new features
to raw-file hashes; `FRESH_CONTRACT.json` records code and environment.

Only this run's checkpoints can be resumed. A changed source, environment,
raw input, or feature bundle fails validation; use a new output directory
after changing code. Checkpoints are stored per session, participant, model
outer fit, post-hoc outer partition, or phase-control dataset. Saved reference
predictions are never substituted for unfinished new fits.

The existing `models`, `exploratory`, `e5`, `posthoc`, and `all` commands retain
their saved-input replay meaning. `raw-features` compares extraction
against saved features; it does not connect those features to the saved-input
model commands. Use **`raw-pipeline`** when verifying the complete new-input
dependency chain. `summarize`, `figures`, and `tables` remain result-rendering
commands and do not certify a fresh raw-to-model run.

Compare newly generated predictions with `audit/`, `results/e5/`, and
`expected/` only in a separate verification process. Preserve the distinction
between preregistered and post-hoc comparisons. Recommended fixed tolerances:
feature `atol=1e-10, rtol=1e-8`; probability/loss absolute error at most `1e-7`;
rosters, labels, nested memberships, selected representations, and C exact.
These tolerances must be declared before a run, not widened to fit its output.

The 20 phase-control datasets reuse the original participants. They evaluate the specified signal controls and do not constitute external-cohort replication. See [known reproduction limits](REPRODUCTION_LIMITS.md) for intermediate-score and calibration differences.
