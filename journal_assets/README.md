# Current journal figures and tables

This presentation layer reproduces the article's five main figures, three
supplementary figures, four main tables, and eleven supplementary tables.
The computational results remain those archived in version 1.2.3
(https://doi.org/10.5281/zenodo.23930239).

From the repository root, using the dependencies in `companion/requirements.txt`:

```text
python -B journal_assets/build.py --output ../eeg-journal-assets
```

Choose a new output directory outside the repository. The command verifies the
SHA-256 identities in `INPUTS.json`, applies explicit presentation changes in
`TABLE_EDITS.json`, and rejects any change to numerical table cells. It draws
Figure 4 directly from the archived group-specific and overall transfer CSVs,
checking that the weighted and equal-class means agree. It does not fit models,
resample participants, or access raw EEG. Missing or changed inputs cause an error.
`VALIDATION.json` and `FILE_MANIFEST.json` identify the resulting displays.

The other seven figure PDFs are copied byte-for-byte from the preserved display
set. Figure 4 uses two side-by-side directions, common method labels, and one
legend. No points, group counts, scales, or scientific comparisons are removed.
Tables use consistent method/device labels and line wrapping without reducing
the main table font. The primary procedure denotes the preregistered logistic
comparison; its selector is named explicitly for exploratory variants.

`INPUTS.json` contains the current figure-number map. Historical numbering in
`companion/` and `docs/technical_assets/` is retained as part of the archive and
must not be used to label current supplementary figures.

To regenerate the historical tables and figures from saved computational
results before applying presentation changes, use the separate documented
`companion/reproduce.py assets` command. That command also recomputes selected
conditional resampling summaries. The current command above uses the archived
numerical table sources, whose values were compared against that regeneration.
Its validation scope is file identity and numerical-cell preservation, not a
complete raw-data or model-training reproduction.

These display updates postdate the v1.2.3 archive. Cite the exact repository
commit for the updated presentation; the v1.2.3 DOI identifies the underlying
archived computations. No new release DOI is assigned by a Git push.

Original Python code is MIT licensed. Original table/figure presentation and
this documentation are CC BY 4.0 under the repository's license notices.
