# Figures and tables

This directory contains the article's figures, tables and the code to regenerate them from saved results. The full article and supplement are maintained separately.

- The code reads inputs directly from `audit/`, `expected/posthoc/` and `results/`; see the [result-file guide](../docs/RESULT_FILES.md).
- `results/numerical_followups/` at the repository root holds the two additional calibration/resampling result files and the resampling specification.
- `figure_source/build_assets.py`: generates figures and tables from saved results.
- `figure_source/REPRODUCIBILITY.md`: analysis settings and formatting conventions.
- `figure_source/ASSET_VALUES.json`: input hashes, displayed values, labels and figure/table numbering.
- `tables/`: eight generated numerical tables, two supplied cohort/flow tables, and the identifier crosswalk (`identifier_crosswalk.tex`). Each file is a LaTeX table fragment.
- `figures/`: main Figures 1–3 and supplementary Figures S1–S3 in PDF/SVG format.

From the repository root:

```text
python -B reproduce.py figures --output ../reproduction
python -B reproduce.py tables --output ../reproduction
```

Both commands write to `<output>/display_assets/` and check values and labels against the supplied records. They combine the saved E3/E4/E5 results and rank participants for plotting in memory, without fitting models. The `tables` command updates tables while keeping the supplied figure files. `figure_source/ASSET_VALUES.json` maps filenames to figure and table numbers.
