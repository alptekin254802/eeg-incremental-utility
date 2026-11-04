# Version 1.2.6

Adds four exploratory follow-ups from dated round-1 plans: saved-prediction
selection decomposition, diagnosis-status entropy/Hjorth, cross-fitted
single-logit EEGNet-like heads, and phase-lag Fourier-phase controls.
The original results remain preserved as reference; no encoder was retrained.

Current journal displays include adjusted baseline losses, diagnosis-status
entropy/Hjorth rows, cross-fitted EEGNet-like heads in the supplement, and
consistent adult-data and Primary procedure labels. The main learner table
omits the original EEGNet row. New cells derive only from saved round-1 outputs.
Technical Methods documents plans, code, outputs and limitations, and replaces
all unsupported operator-name commands with upright math text for GitHub rendering.

Validation covers manifest identities, saved-result consistency, current display
regeneration, clean-source builds, and existing companion portability tests.
No new predictive fit or raw-data extraction is performed by this release step.
