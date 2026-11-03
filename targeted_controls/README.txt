Targeted sensitivity addendum - 3 November 2026

This local manuscript addendum evaluates three bounded exploratory questions:
an oracle prior-odds diagnostic of device transfer; flexible age in matched
models; and one additional 30 Hz low-pass filter before entropy/Hjorth extraction.
The original primary increment remains unfavorable. Candidate entropy/Hjorth
gains have smaller point estimates in the new sensitivities. Prior correction
reverses three of four transfer increments. These are sensitivity results,
not physiological identification or an externally validated clinical model.

The plan in results/ANALYSIS_PLAN.txt was recorded before these new outcomes,
after earlier outcomes were known. This is not an original preregistered study.
All five refitted comparisons and all four offset comparisons are retained.
No new DOI is assigned to this separate addendum. It is not contained in the
published 1.2.1 archive, whose content remains unchanged.

DEPENDENCIES
Obtain the published computational companion 1.2.1:
https://doi.org/10.5281/zenodo.23864656
Set RELEASE to its companion/reference_release directory and TRANSFER to
companion/extensions/device_transfer. Exact input identities are checked.
The version 1.1.0 reference release has the same scientific dependencies.
Install Python dependencies: python -m pip install -r requirements.txt
Recorded environment: Python 3.14.3; pinned package versions below.

QUICK VERIFICATION (no predictive fits, no raw EEG required)
From this addendum directory, replace uppercase directory arguments:
python -B code/run.py verify --release RELEASE --transfer TRANSFER --output NEW_VERIFY_DIR
This verifies all bundled/dependency file identities, reproduces the original
transfer scores and new oracle probabilities, replays all 5,000-draw conditional
summaries, scores the saved nested predictions, checks all tuning selections,
and regenerates both supplementary tables. It compares all CSV values and
generated tables with supplied outputs. Use a new output directory every time.
It does not refit the age/bandwidth models or reconstruct raw features.

FULL TARGETED RECOMPUTATION (not a rerun of the entire historical project)
python -B code/run.py init --release RELEASE --transfer TRANSFER --output NEW_RUN_DIR
python -B code/run.py age --release RELEASE --transfer TRANSFER --output NEW_RUN_DIR --workers 4
python -B code/run.py extract --release RELEASE --transfer TRANSFER --output NEW_RUN_DIR --raw RAW_DATASET_DIR --workers 2
python -B code/run.py bandwidth --release RELEASE --transfer TRANSFER --output NEW_RUN_DIR --workers 4
python -B code/run.py offset --release RELEASE --transfer TRANSFER --output NEW_RUN_DIR
python -B code/run.py summarize --release RELEASE --transfer TRANSFER --output NEW_RUN_DIR

RAW_DATASET_DIR is the extracted BALLADEER dataset directory containing the
relative files in results/extract/RAW_RECONSTRUCTION.json. Obtain the source
through https://doi.org/10.6084/m9.figshare.28676042 and retain its own terms.
No raw EEG is redistributed. The reconstruction checks all 96 raw file hashes,
accepted windows, quality summaries, and original spectral and entropy/Hjorth
features. An input mismatch fails explicitly. Saved derived filtered features
are bundled; they are not interchangeable with features from a different
preprocessing/version. Outputs are always separate from archived inputs.

RESULT FILES
results/age and results/bandwidth retain every inner score, regularization
choice, outer prediction, candidate selector result, and per-fold job record.
results/extract retains the nine filtered features, feature comparisons, raw
identity and accepted-window provenance. Participant IDs are source pseudonyms.
results/offset retains original/oracle paired probabilities, all metrics,
class-specific and equal-class differences, and every conditional draw.
results/summary retains five paired estimates, repeat and participant results,
changes relative to originals, all conditional draws, and numerical checks.
tables/ contains manuscript Tables S19 and S20 and full-precision input values.
The original figures and original estimates are unchanged.

VALIDATION SCOPE
All planned sensitivity fits completed: 450 new outer models plus inner tuning,
and three original first-fold reference replays. Raw feature reconstruction
covered 96 participants. This is full execution of this limited analysis family,
not raw-to-end reproduction of every earlier analysis and not an external test.
Uncertainty ranges condition on fixed predictions; they omit fitting/selection
uncertainty. Oracle offsets use target labels and cannot be deployed as assessed.
The 30 Hz sensitivity adds a filter to existing 1-40 Hz preprocessing and retains
all original accepted windows. Age and bandwidth changes are separate analyses.

LICENSE AND CITATION
Original code: MIT (LICENSE_CODE.txt). Original documentation, derived results,
and tables: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/ .
Copyright 2026 Alptekin Cinbat, Umit Senturk, Ibrahim Yucedag.
Third-party and source-data rights are not replaced by these licenses.
The code calls the preserved original reference release; its MIT notices remain
in that dependency. This addendum does not redistribute third-party papers.
Credit the study authors, this dated targeted sensitivity addendum, the published
companion version actually used, and the source dataset. No publication date
or DOI for this manuscript or addendum is asserted.
