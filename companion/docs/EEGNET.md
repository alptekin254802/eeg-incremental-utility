# Compact learned-representation comparison

This is one exploratory EEGNet-like representation pipeline, planned after the
earlier manuscript results were known. It is not an extension of the original
preregistration, a new architecture, or a state-of-the-art benchmark. The network
uses the temporal/depthwise/separable structure of Lawhern et al. (2018),
doi:10.1088/1741-2552/aace8c, independently implemented in PyTorch.

The original 96 participants and all 50 outer/250 inner partitions are preserved.
Each of 300 encoders learns only from its training participants. Participant bags
and mean embeddings preserve the participant as the analysis unit. A regularized
logistic head combines embeddings with the same seven baseline variables. The
comparison changes the EEG representation while preserving the paired-addition
question. The final model is not a jointly trained multimodal neural network.

## Dedicated environment

The EEGNet reference run uses Python 3.12.14, PyTorch 2.5.1+cu118, NumPy 1.26.4,
SciPy 1.16.3, pandas 2.3.3, scikit-learn 1.8.0, and matplotlib 3.10.7.
Use a separate environment outside the repository. The full recorded environment
is in `extensions/eegnet/ENVIRONMENT.txt`. CPU execution is supported; numerical
identity across operating systems, devices, and library versions is not promised.
GPU execution requires a compatible NVIDIA driver. No remote service is used.

```text
python -m venv ../eegnet-env
../eegnet-env/Scripts/python -m pip install -r extensions/eegnet/ENVIRONMENT.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

On POSIX systems, replace `Scripts/python` with `bin/python`. The first command
must use a Python 3.12 interpreter. For CPU-only execution, install the direct
scientific dependencies from `extensions/eegnet/requirements.txt` and PyTorch
2.5.1 from its official CPU wheel index instead of the recorded CUDA environment.
The frozen preprocessing uses `numpy.trapezoid`,
unavailable in the training environment's NumPy 1.26.4. Reconstruct windows in the
main Python 3.14.3 / NumPy 2.3.4 environment specified by the root requirements.
This preserves the original preprocessing code. Saved-result verification and
asset generation also use that main environment without PyTorch.

## Commands and inputs

In the main scientific environment:

```text
python -B eegnet.py prepare --dataset /path/to/BALLADEER/dataset --output ../eegnet-preparation
python -B eegnet.py verify --output ../eegnet-saved-check
```

In the dedicated EEGNet environment:

```text
python -B eegnet.py synthetic-check --device cuda --output ../eegnet-synthetic
python -B eegnet.py train --cache ../eegnet-preparation/windows --device cuda --output ../eegnet-training
python -B eegnet.py replay --cache ../eegnet-preparation/windows --device cuda --output ../eegnet-selected-replay
```

`prepare` reads existing raw EEG from the public BALLADEER dataset; it does not
download data. It reconstructs accepted windows and checks frozen quality measures
and spectral features. The raw root must contain the participant directories and
`users_demographics.json`. All raw and cached files remain unchanged by training.
The cache identity includes source hashes, preprocessing-code identities, participant
roster, dimensions, and actual array hashes. Timestamps are excluded from the data
fingerprint so a byte-identical fresh reconstruction can be verified.

`train` performs all 300 planned encoders and all nested logistic heads. `replay`
performs just the six encoders and heads for repeat 0 / outer fold 0. Neither
automatically changes the configuration after observing performance. Outputs must
be new and outside source/data/cache locations; partial outputs are retained, and
no resume or overwrite is performed. A missing input or identity mismatch is an
explicit error. Package identity is checked before and after each command.

`verify` checks all saved nested memberships, candidate scores, logistic-head
probabilities, frozen baseline identities and repeat metrics; it regenerates the
10,000 fixed-prediction participant-bootstrap draws and the EEGNet display.
This does not retrain encoders. The intervals condition on the recorded predictions
and exclude training, initialization, and selection uncertainty.

## Retained material

`ANALYSIS_PLAN.txt` and `PLAN_LOCK.json` precede clinical fitting. `code/` contains
the independent implementation, preprocessing adapter, synthetic checks, and
saved-result verifier. `results/` retains all 300 weight checkpoints, embeddings,
normalization parameters, training curves, candidate heads/scores and 960 paired
predictions. Checkpoints contain tensors only; load with `weights_only=True`.
`reports/` contains all repeat metrics and conditional draws. Raw EEG/window arrays
are not redistributed here; reconstruct them from the attributed source data.

Original code in this extension is MIT licensed. Original documentation, figures,
tables and derived outputs are CC BY 4.0 under the root notices. The BALLADEER data
and historical reference release retain their source attribution and licenses.
No source from the authors' EEGNet software repository is redistributed.
