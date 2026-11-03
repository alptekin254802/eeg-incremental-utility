"""Reproduce the current journal displays without modifying archived computations.

Tables retain their verified archived numerical cells. Explicit editorial rules
update labels and layout; Figure 4 is drawn directly from the saved CSV results.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
from build_figure4 import draw

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numeric_cells(text):
    """Numeric content after row labels, excluding LaTeX layout dimensions."""
    rows = [line.split("&", 1)[1] for line in text.splitlines()
            if "&" in line and not line.lstrip().startswith((r"\caption", "%"))]
    return re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", "\n".join(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output == ROOT or output in ROOT.parents or ROOT in output.parents:
        raise ValueError("Choose an output directory outside the repository")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output must be new or empty; existing files are never replaced")
    config = json.loads((HERE / "INPUTS.json").read_text(encoding="utf-8"))
    for name, expected in config["inputs"].items():
        path = ROOT / name
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"Missing or changed scientific input: {name}")
    rules = json.loads((HERE / "TABLE_EDITS.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    (output / "tables").mkdir()
    (output / "figures").mkdir()
    verified = {}
    for name, source in config["tables"].items():
        original = (ROOT / source).read_text(encoding="utf-8")
        edited = original
        for before, after in rules.get(name, []):
            if edited.count(before) != 1:
                raise ValueError(f"Editorial rule does not match exactly once: {name}")
            edited = edited.replace(before, after, 1)
        if numeric_cells(original) != numeric_cells(edited):
            raise ValueError(f"Editorial rule changed numerical cells: {name}")
        (output / "tables" / name).write_text(edited, encoding="utf-8")
        verified[name] = {"numeric_cells_unchanged": True, "numeric_tokens": len(numeric_cells(edited))}
    for name, source in config["figures"].items():
        if name != "device_transfer.pdf":
            shutil.copyfile(ROOT / source, output / "figures" / name)
    figure = draw(ROOT / config["transfer_results"], output / "figures")
    report = {"status": "PASS", "scope": "File identities, numerical-cell preservation, and figure generation from saved results",
              "input_count": len(config["inputs"]), "tables": verified, "figure4": figure,
              "figure_numbering": config["figure_numbering"], "new_fits": 0, "resampling_runs": 0}
    (output / "VALIDATION.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    manifest = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}
    (output / "FILE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {len(verified)} tables, {len(config['figures'])} figures; numerical cells preserved.")


if __name__ == "__main__":
    main()
