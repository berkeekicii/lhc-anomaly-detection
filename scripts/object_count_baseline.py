"""How much of the separation comes from simply counting objects?

Usage (from the project root, after a run has been exported):

    python scripts/object_count_baseline.py results/long

The autoencoder sees 57 numbers per event, but signal events also contain more
objects than background events. This script measures the AUC of a trivial
"detector" that uses nothing but the number of jets (or electrons, or muons)
in an event, so that the autoencoder's AUC can be compared against it.

The counts are read from curves.json, which lhcad.export writes, so the
dataset itself is not needed.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

from lhcad.constants import DATASETS, SIGNALS


def auc_from_histograms(background, signal) -> float:
    """AUC of a discrete variable from its count histograms; ties count half."""
    b = np.asarray(background, dtype=float)
    s = np.asarray(signal, dtype=float)
    b, s = b / b.sum(), s / s.sum()
    n = max(len(b), len(s))
    b, s = np.pad(b, (0, n - len(b))), np.pad(s, (0, n - len(s)))
    greater = sum(s[i] * b[:i].sum() for i in range(n))
    return float(greater + 0.5 * (s * b).sum())


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 1
    folder = Path(argv[0])
    curves = json.loads((folder / "curves.json").read_text())
    if "inputs" not in curves:
        print(f"ERROR: {folder}/curves.json has no input statistics "
              "(export the run with --raw-dir pointing at the data).", file=sys.stderr)
        return 1
    files = curves["inputs"]["files"]
    metrics = json.loads((folder / "metrics.json").read_text())

    objects = ("jet", "electron", "muon")
    print(f"AUC of counting one kind of object, next to the autoencoder of run '{metrics['run']}'\n")
    print(f"{'signal':<16}" + "".join(f"{o + 's':>14}" for o in objects) + f"{'autoencoder':>14}")
    for name in SIGNALS:
        if name not in files or name not in metrics["signals"]:
            continue
        cells = "".join(
            f"{auc_from_histograms(files['background']['objects'][o]['count_histogram'], files[name]['objects'][o]['count_histogram']):>14.3f}"
            for o in objects)
        print(f"{DATASETS[name]['label']:<16}{cells}{metrics['signals'][name]['auc']:>14.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
