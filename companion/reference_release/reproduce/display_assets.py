"""Generate the article's figures and tables from saved results in a separate directory."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables-only', action='store_true')
    args = parser.parse_args()
    release = Path(os.environ['EEG_RELEASE_ROOT'])
    output = Path(os.environ['EEG_RUN_ROOT']) / 'display_assets'
    source = release / 'display_assets'
    assert source.is_dir(), 'The release must include the display_assets directory'
    # Read results from the repository and write figures, tables and recorded values.
    command = [sys.executable, '-B', str(source / 'figure_source/build_assets.py'),
               '--release-root', str(release), '--output', str(output)]
    if args.tables_only:
        command.append('--tables-only')
    subprocess.run(command, check=True)
    reference = json.loads((source / 'figure_source/ASSET_VALUES.json').read_text(encoding='utf-8'))
    generated = json.loads((output / 'figure_source/ASSET_VALUES.json').read_text(encoding='utf-8'))
    assert generated == reference, 'Article display values, labels, or source identities changed'
    record = dict(mode='saved-result display regeneration; no fitting',
                  display_values_and_labels_equal=True,
                  reference_sha256=hashlib.sha256((source/'figure_source/ASSET_VALUES.json').read_bytes()).hexdigest(),
                  locations=generated['display_locations'])
    (output / 'DISPLAY_CHECKS.json').write_text(json.dumps(record, indent=2) + '\n')
    print('Current article displays verified in display_assets/; fresh analysis outputs were not read or replaced.')


if __name__ == '__main__':
    main()
