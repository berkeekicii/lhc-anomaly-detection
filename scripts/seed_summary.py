"""Average several runs of the same setting that differ only in the random seed.

Usage (from the project root):

    python scripts/seed_summary.py results/long-seed*

A single run tells you what one random initialisation produced. Repeating the
same setting with different seeds shows how much of a difference between two
settings is real and how much is noise. This prints, per signal, the mean AUC
over the runs and the spread (sample standard deviation), plus the smallest
and largest value seen.
"""

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

from lhcad.constants import DATASETS, SIGNALS


def main(argv: list[str]) -> int:
    folders = [Path(a) for a in argv]
    missing = [f for f in folders if not (f / "metrics.json").exists()]
    if len(folders) < 2 or missing:
        print(__doc__, file=sys.stderr)
        if missing:
            print(f"no metrics.json in: {', '.join(str(f) for f in missing)}", file=sys.stderr)
        return 1
    runs = [json.loads((f / "metrics.json").read_text()) for f in folders]
    seeds = []
    for folder in folders:
        config = folder / "train_config.json"
        seeds.append(json.loads(config.read_text()).get("seed") if config.exists() else None)
    print(f"{len(runs)} runs: " + ", ".join(
        f"{r['run']}" + (f" (seed {s})" if s is not None else "") for r, s in zip(runs, seeds)))

    print(f"\n{'signal':<16}{'mean AUC':>10}{'std':>9}{'min':>9}{'max':>9}")
    for name in SIGNALS:
        values = [r["signals"][name]["auc"] for r in runs if name in r["signals"]]
        if len(values) != len(runs):
            continue
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        print(f"{DATASETS[name]['label']:<16}{statistics.fmean(values):>10.4f}{std:>9.4f}"
              f"{min(values):>9.4f}{max(values):>9.4f}")
    print("\nA difference between two settings is only meaningful if it is larger than this spread.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
