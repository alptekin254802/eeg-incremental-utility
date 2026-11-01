# Data layout

Obtain BALLADEER from the [dataset record](https://doi.org/10.6084/m9.figshare.28676042). Respect the dataset's own access and reuse terms.

Pass the extracted directory containing `users_demographics.json` as `--dataset`. It must also contain participant folders such as `UB0004/AttentionRobotsDesktop/<session>/`, with Emotiv EPOC CSV, EYE_TRACKING_DATA and GAME_DATA files. The command discovers all complete sessions and checks the expected roster; changes to the data release can fail these checks.

```powershell
python reproduce.py raw-screen --dataset "D:/data/BALLADEER/dataset" --output ../reproduction
```

No Windows junction or hard-coded drive is required. Archived Windows path separators are normalized by the feature and signal-control wrappers. The output directory must be outside both this release and the dataset. Source recordings are opened for reading only. Reproduction outputs contain local path provenance and participant-level derived values; review the dataset's terms before redistributing those outputs.

This package has no automated data download step. An independently identified, immutable dataset export and clean environment remain part of final release verification.
