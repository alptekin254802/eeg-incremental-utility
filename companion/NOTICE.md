# Attribution and license scope

Original analysis code, reproduction wrappers, figure/table generators and tests
are available under the [MIT License](LICENSES/MIT.txt). Original documentation,
figures, tables and derived scientific content are available under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). These grants cover the
contributors' rights only. Dependencies are not bundled or relicensed.

## BALLADEER / Attention Robots

The source dataset is **A Multimodal Dataset for Neurophysiological and AI
Applications**, by Juan Trujillo, Rosario Ferrer-Cascales, Miguel A. Teruel,
Nicolás Ruiz-Robledillo, Javier Sanchis, Sandra García-Ponsoda, Alejandro
Panagiotidis, Natalia Albaladejo-Blázquez, Ángela Martínez-Nicolás, Jorge
García-Carrasco, Alejandro Reina, Ana Lavalle, Alejandro Maté, and Borja Costa-López.
Dataset: [Figshare 28676042](https://doi.org/10.6084/m9.figshare.28676042), provider
versions 1 and 2, CC BY 4.0. Descriptor:
[10.1038/s41597-026-06758-7](https://doi.org/10.1038/s41597-026-06758-7).

Raw EEG is obtained separately. The derived participant features, pseudonymous
predictions, and summaries under `reference_release/`, `extensions/` and
`article_assets/` retain source attribution and applicable CC BY 4.0 terms.
Transformations and exclusions are documented in the Methods and executable
analysis specifications. Derived content is not an unmodified source dataset.

## Independent adult data and third-party scripts

The unchanged files under `extensions/utility/external/` come from
[OSF EWKGV](https://doi.org/10.17605/OSF.IO/EWKGV). The source study is Annika Wiebe,
Benjamin Selaskowski, Martha Paskin, Laura Asche, Julian Pakos, Behrem Aslan, Silke
Lux, Alexandra Philipsen, and Niclas Braun, *Virtual reality-assisted prediction
of adult ADHD based on eye tracking, EEG, actigraphy and behavioral indices:
a machine learning analysis of independent training and test samples*,
Translational Psychiatry 14, 508 (2024),
[10.1038/s41398-024-03217-y](https://doi.org/10.1038/s41398-024-03217-y).

Those source CSVs, coding guide, and original source-study scripts retain their
CC BY 4.0 license; they are **not** relicensed under MIT. Their source URLs and
SHA-256 digests are in `DOWNLOAD_MANIFEST.json`; provider metadata and full license
text are preserved alongside them. The copy in `LICENSES/CC-BY-4.0.txt` is extracted
from that preserved license metadata. The adult reanalysis outputs are new derived
content, with source data bytes unchanged. No full journal articles are bundled.

## Historical and current material

`reference_release/` preserves version 1.1.0, including its original citation and
license notices. The locked plans in `extensions/` are preserved historical
scientific documents. They have not been rewritten to make later work appear
preregistered. New packaging and display code are identified separately.
The new candidate does not inherit a publication date or version DOI from the
older archive. Journal manuscript files and declarations are distributed in the
separate submission package, outside this software repository.

## EEGNet architecture attribution

The new PyTorch implementation follows the compact architectural principles of Lawhern et al., *EEGNet*, Journal of Neural Engineering 15 (2018) 056013, doi:10.1088/1741-2552/aace8c. It adapts participant-level training and multimodal prediction to this study. No upstream EEGNet software source, pretrained weights, or paper PDF is bundled. The independent new code retains the root MIT license.
