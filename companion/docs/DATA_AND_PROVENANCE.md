# Data and provenance

## Pediatric source

[BALLADEER / Attention Robots](https://doi.org/10.6084/m9.figshare.28676042) is the
source, described in [Scientific Data](https://doi.org/10.1038/s41597-026-06758-7).
Provider versions 1 and 2 state CC BY 4.0 and point to the identical archive: file ID 53280374, 13,633,870,282 bytes, provider MD5 c7e7b5d57855ada4a380f667d1a5cf85. Thus those two versions do not establish different raw archives. The local metadata and README still match their analysis-stage SHA-256 records. A complete local-to-provider archive identity check remains unperformed because the original downloaded archive was not located; the local roster differs from the broader descriptor. The README version 1.0 is an internal document version, not a verified Figshare version mapping.

The saved screening denominator is 126 readable multimodal sessions, followed by
98 measurement-eligible and 96 EEG-eligible participants. The final 96-person
roster, feature/QC inputs, target linkage, and covariance arrays have explicit
SHA-256 checks in `reference_release/`. GROUP is retained as supplied; discordant
diagnosis entries are not silently recoded. The diagnosis-status sensitivity
contains 84 participants and changes both target and sample.

Raw EEG is not bundled. For raw reproduction, obtain the data from its provider
and supply a directory containing `users_demographics.json` and participant task
folders in the layout described by `reference_release/docs/DATA_LAYOUT.md`.
Read-only data access and roster/hash checks are required. A mismatched export is
an error, not permission to drop participants or bypass the checks.

## Adult source

`extensions/utility/external/DOWNLOAD_MANIFEST.json` records six unchanged source
files, URLs and SHA-256 identities from [OSF EWKGV](https://doi.org/10.17605/OSF.IO/EWKGV).
Source-study attribution and CC BY 4.0 notices are preserved. Released row numbers
are the available identifiers. The original 50/36 split is retained, without
pooling adult and pediatric participants. Exact-row duplicate checks do not
independently verify recruitment identities.

## Derived and historical material

The full `reference_release/` preserves the 145 supplied v1.1.0 files. Later plans,
source code and saved results remain separate in `extensions/`. File manifests
provide identity, not proof that a file was attached to the original registration.
Historical execution code hashes can differ from distributed portable code;
this distinction must be traced, not hidden by updating historical hashes.

Synthetic archives omit only duplicated covariates `test_B`, `test_E`, `test_D`
and `test_shiftD`. The fixed generator reconstructs them; all labels, fitted and
oracle probabilities remain. `SYNTHETIC_ARRAY_PACKAGING.json` records that
representation change and original hashes. It does not change simulated results.

The historical diagnosis-sensitivity archive lacks participant predictions and
selected regularization values. Its repeat-level metrics are available. New
predictions can be generated, but cannot be asserted byte-identical to a missing
historical file. These limits are separate from the closed spectral-state and
scalar-calibration corrections documented in the reference release.

## Source-study clinical procedures

The descriptor reports clinician-based DSM-5 assessment, exclusion of severe comorbid presentations, and instructions for at least a 24-hour medication washout monitored by the clinical team. Subtype was not systematically recorded. Individual medication type/dose, adherence and comorbidity details are absent from the linked analysis inputs; protocol reporting does not establish those individual values or explain discordant GROUP/diagnosed entries. The saved registration explicitly documented the discrepancy and unanswered author contact before fixing the primary target and sensitivity. See SOURCE_DOCUMENTATION_CHECK.md for the recheck scope.

Three path-only metadata fields in later addenda are normalized for public distribution. `METADATA_PATH_NORMALIZATION.json` records the original/distributed hashes, changed keys and package-relative meanings. These are labeled distribution derivatives; the original plans, data hashes and scientific fields are preserved. The nested package manifest identifies the distributed bytes.
