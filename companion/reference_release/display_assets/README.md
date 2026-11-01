# Figures and tables

This directory contains code to generate the article's figures and tables from saved records. Generated tables and figures are written outside the repository. The full article and supplement are maintained separately.

- The code reads inputs directly from `audit/`, `expected/posthoc/` and `results/`; see the [result-file guide](../docs/RESULT_FILES.md).
- `results/numerical_followups/` at the repository root holds the two additional calibration/resampling result files and the resampling specification.
- `figure_source/build_assets.py`: generates figures and tables from saved results.
- `figure_source/REPRODUCIBILITY.md`: analysis settings and formatting conventions.
- `figure_source/ASSET_VALUES.json`: input hashes, displayed values, labels and figure/table numbering.
The outputs comprise eleven LaTeX table fragments and six figures in PDF/SVG format. The cohort tables use the saved participant, eligibility and quality-control records. The identifier crosswalk uses the method definitions in the builder.

From the repository root:

```text
python -B reproduce.py figures --output ../reproduction
python -B reproduce.py tables --output ../reproduction
```

Both commands write to `<output>/display_assets/` and check values and labels against the supplied records. They combine the saved E3/E4/E5 results and rank participants for plotting in memory, without fitting models. The `tables` command generates all tables without writing figures. `figure_source/ASSET_VALUES.json` maps filenames to figure and table numbers.
