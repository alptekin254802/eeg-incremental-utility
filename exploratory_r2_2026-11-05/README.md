# Exploratory round-2 work, 2026-11-05

> Version 1.2.6 note: this is the dated round-1 execution record. Its four follow-ups are now documented in Supplementary Sections S3, S4 and S9 and docs/TECHNICAL_METHODS.md. Executed analysis code and numerical outputs are preserved.

Local revision work; no release or remote push. Scientific archived inputs are unchanged.
All new computations and outputs are in this directory. See ANALYSIS_REPORT.md.
Do not insert the diagnosis, cross-fitting or phase-lag results into the article in round 1.

Run the plans before repeating any fit. Main commands from the repository root:

```
python -B exploratory_r2_2026-11-05/selection/audit_selection.py
python -B exploratory_r2_2026-11-05/selection/verify_with_sklearn.py
python -B exploratory_r2_2026-11-05/recorded_manuscript_audit.py
python -B exploratory_r2_2026-11-05/diagnosis/run_diagnosis.py
python -B exploratory_r2_2026-11-05/eegnet_crossfit/infer_saved_encoders.py
python -B exploratory_r2_2026-11-05/eegnet_crossfit/evaluate_heads.py
python -B exploratory_r2_2026-11-05/phase_lag/extract_phase_lag.py
python -B exploratory_r2_2026-11-05/phase_lag/evaluate_phase_lag.py
python -B exploratory_r2_2026-11-05/redraw_figures.py
```

Frozen-encoder inference requires CPU PyTorch 2.5.1 with NumPy 1.26.4;
it was run using F:/proje/eeg_dl_thesis_lab/.venv/Scripts/python.exe.
Head evaluation and other analyses use the repository's NumPy 2.3.4,
pandas 2.3.3, SciPy 1.16.3, scikit-learn 1.8.0 environment (Python 3.14).
Inference requires the verified float32 window cache referenced in its script;
phase reconstruction requires the raw BALLADEER path referenced in its script.
No cache/raw files are written. Checkpointed feature/inference outputs can be reused.
Every plan and result provenance includes source identities. Conditional ranges
hold models and predictions fixed and omit training/model-selection uncertainty.
