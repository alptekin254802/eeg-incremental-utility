# Current journal displays, version 1.2.6

```
python -B journal_assets/build.py --output ../eeg-journal-assets
```

Choose a new output outside the repository. INPUTS.json checks every frozen
input; TABLE_EDITS.json records exact old/new table presentation. Four table
updates intentionally change numerical cells: matched adjustment baselines,
omission of the original main-text EEGNet row, diagnosis entropy/Hjorth, and
cross-fitted EEGNet-like heads. These cells are checked against round-1 CSVs.
All other numerical cells are preserved.

Figures 1, 2 and S1 are redrawn from recorded results with the Primary procedure
label and checked against archived plot records; Figure 4 is redrawn from group
and aggregate transfer CSVs. Other figures retain their archived bytes.
Fixed PDF metadata makes repeat builds byte-identical. The command does not
fit models, resample participants or access raw EEG. VALIDATION.json and
FILE_MANIFEST.json identify all 16 current tables and eight figures.
Historical companion/docs assets retain their own numbering and values.
Original code is MIT licensed; original documentation and displays are CC BY 4.0.
