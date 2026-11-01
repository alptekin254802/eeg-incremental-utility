# Participant flow and computational provenance

This retrospective descriptive audit used preserved screening tables, preprocessing records, source code and saved results. Its analysis plan was recorded before calculating the additional retention summaries. No raw signals were opened, features re-extracted or predictive models fitted. The records support the stated **126 → 98 → 96** participant flow and identify the historical source files corresponding to the recorded code hashes. They do not establish an immutable raw-dataset export identity.

## Participant flow

The readable multimodal candidate manifest contains 126 sessions from 126 distinct participants. This denominator is separate from the larger metadata roster and the modality-specific quality tables. EEG, gaze and behavioral audit tables have 132, 145 and 150 rows, respectively; their row counts are not alternative denominators for the complete multimodal cohort.

| Stage | Retained | Excluded at stage |
|---|---:|---:|
| Readable EEG, object-gaze and behavioral candidate records | 126 | 0 |
| Measurement eligibility | 98 | 28 |
| EEG preprocessing | 96 | 2 |

Eligibility was reconstructed from recorded metrics using the archived predicates. EEG and gaze duration each had to be at least 240 seconds; EEG nonfinite fraction could not exceed 0.05, maximum EEG timestamp gap could not exceed 2 seconds, and gaze valid-row fraction had to be at least 0.80. The code additionally required the recorded readable/structural flags and absence of specified timestamp or flat-channel faults. No longest-gaze-gap exclusion was applied. The full rule list and failure counts are in `SCREEN_RULE_FAILURES.csv`.

Among the 28 measurement exclusions, seven failed only the EEG gap rule, 19 failed only the gaze-duration rule and two failed both. Thus nine exceeded the EEG gap limit and 21 had gaze recordings shorter than 240 seconds. No other measurement predicate excluded a member of this 126-person candidate cohort. The subsequent EEG preprocessing exclusions were one participant with more than two bad channels and one with less than 120 seconds of accepted clean EEG. Both were in the Control group and used the Epoc+ device. The reconstructed 98-person roster exactly matches the saved measurement cohort; the 96-person roster exactly matches the saved feature matrix.

All 126 candidate records contain age, the source's sex field, group, diagnosed status and device model. The retained participants' age, sex, group and diagnosed values also match the released outcome-linkage table. These are observed metadata values, not newly adjudicated clinical labels.

| Characteristic | Retained (n=96) | Excluded (n=30) |
|---|---:|---:|
| Age, mean (sample SD), years | 12.20 (3.58) | 13.27 (3.88) |
| Age range, years | 6–18 | 6–18 |
| Experimental / Control group | 41 / 55 | 10 / 20 |
| Source sex: Female / Male | 36 / 60 | 15 / 15 |
| Device: Epoc X / Epoc+ | 54 / 42 | 13 / 17 |
| Diagnosed: yes / no / undetermined | 54 / 30 / 12 | 18 / 8 / 4 |

These summaries describe observed attrition; they do not demonstrate absence of selection bias. No significance tests or adjusted retention models were run. `RETENTION_PROFILE.csv` also gives the 126-person starting cohort and each exclusion stage. `RETENTION_BY_GROUP_DEVICE.csv` provides aggregate group-by-device counts. A minimal eligibility ledger is supplied without the expanded participant-level demographic profile. Aggregate tables describe the retained and excluded groups.

## Historical sources and portable release

All five available historical source files have SHA-256 digests identical to their corresponding historical provenance records. Their public portable counterparts have different byte hashes. `CODE_SOURCE_MAP.csv` explicitly identifies both hashes, the provenance record and its relevant field; a current portable hash must not be presented as the original executed-source hash.

| Module | Identical retained function/class ASTs | Changes inspected |
|---|---:|---|
| Screening | 13/13 | Output location renamed; metadata/report/orchestration functions removed from the portable parser module. Public screening wrapper reuses its parsers and the measurement-eligibility function. |
| Stage 1 extraction | 27/28 | The remaining function changes its audit directory, messages and summary-key names; eligibility thresholds are unchanged. Extraction/report entry points were removed from the library module. |
| Stage 1.5 representations | 21/21 | Historical reporting/orchestration functions and a freeze-document dependency removed. |
| Primary analysis | 31/33 | Expected hashes updated for portable code and sanitized provenance; private document dependencies removed. One fold-provenance descriptive field removed. Split, transformation, estimator and tuning functions are unchanged in the compared common functions. |
| Exploratory battery | 20/20 | Historical report/plan/execution entry points removed; the shared extraction and modeling functions are unchanged. |

The common top-level assignments also retain the scientific constants; the only changed common assignment is the screening output location. Removed assignments point to historical documentation. These are static comparisons, not proof that every historical and portable execution path is equivalent. The portable wrappers relocate orchestration and output handling. The evidence includes common-function AST digests, names of removed functions, top-level assignment comparisons and focused text diffs. Complete original scripts are not republished here; preserved historical files were inspected to verify their hashes, and original and portable sources remain distinctly identified.

**E5 is a separately documented scientific correction.** The historical raw feature table is ordered by region, so contiguous groups of nine columns correspond to regions. The later E5 wrapper explicitly reorders the same 27 variables by frequency band before the unchanged blockwise PCA function is called. This changes PCA grouping and is not merely a file-path or packaging change. It is already incorporated in the current saved E5 analysis and manuscript: the saved band-ordered table contains the same feature values after column alignment, the saved specification has nine variables per band, and the current manuscript describes band-specific PCA. The saved mean log loss 0.6336906 and difference −0.0093113 round to the manuscript's 0.6337 and −0.0093. `E5_GROUPING_MAP.csv` and `E5_CURRENT_RESULT_CHECK.json` document these checks. No current mismatch was found, and no E5 model was rerun in this audit.

## Identity and verification limits

`INPUT_SHA256.csv` records hashes calculated from every evidence file used in this audit. `RECORDED_SOURCE_DIGESTS.json` separately reports upstream dataset-file hashes copied from historical provenance; those raw files were not independently rehashed in this follow-up. Neither set of hashes establishes a versioned, immutable raw-data export or authenticates when a historical run occurred. A persistent dataset version/export identifier and an associated complete raw-file manifest would be needed to close that identity gap. This audit also does not independently validate the clinical group labels or the raw measurements.

Checks actually executed were table-key and join validation; eligibility reconstruction from saved metrics; exact saved-roster and retained-metadata comparisons; descriptive retention aggregation; source-byte hashing; static function/class and top-level assignment comparisons; and saved E5 feature-order/specification/table consistency checks. Screening, preprocessing, feature-extraction and prediction programs were inspected but not executed. The checks do not constitute a fresh raw-data or model reproduction.

## Clinical metadata and adult protocol

Medication status and psychiatric comorbidity were not available in the participant-level pediatric inputs used for this analysis. This does not establish that the source study never collected them. Consequently, age/sex adjustment cannot establish independence from medication, comorbidity or other unmeasured clinical heterogeneity.

For the adult comparator, primary-source Methods text reports that training-set ADHD participants were not currently receiving medication, whereas test-set ADHD participants stopped medication for 48 hours. Training EEG powers were averaged across three 18-minute task blocks; the test analysis used only the single no-feedback block from a feedback experiment. These differences constrain the adult result's interpretation as a transport check. The full text was available and the relevant Methods passages were inspected; these statements are not inferred from an abstract. [Wiebe et al. (2024)](https://www.nature.com/articles/s41398-024-03217-y).
