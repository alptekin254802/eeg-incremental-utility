# Version 1.2.5

This update aligns the computational companion with the current article and
supplement without changing models, predictions, random seeds, or numerical results.

- Reproduces all 16 current journal tables, including the evaluation workflow.
- Maps learning curves to main Figure 5, the adult comparison to Figure 6,
  and simulations to supplementary Figure S2; all eight figure PDFs are retained.
- Aligns the Technical Methods note's terminology and section/table references;
  retains implementation details removed from the condensed supplement.
- Removes two untracked Python cache files from the release manifest so a clean
  checkout can pass the documented manifest check.

The manuscript and supplement are maintained separately from this software
repository. Article page placement is a LaTeX layout choice and does not alter
the scientific figure files. Historical numerical assets remain unchanged.

Validation covers file identities, current table/figure generation and saved
measurement checks. No new model fitting or raw-data reconstruction is required
by this documentation and presentation update.
