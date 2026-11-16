"""Look inside a raw ADC2021 file: structure, one example event, basic statistics.

Usage (from the project root):

    python -m lhcad.explore data/raw/background_for_training.h5
    python -m lhcad.explore data/raw/Ato4l_lepFilter_13TeV_filtered.h5 --events 50000

Prints a summary and saves a figure to figures/explore_<file name>.png .
Run it on the background file and on a signal file and compare the two:
that comparison is the reason anomaly detection can work at all.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .constants import CLASS_INDEX, OBJECT_SLOTS, PARTICLES_KEY
from .data import _particles, describe_file


def read_raw(path, n_events: int) -> tuple[np.ndarray, int]:
    """First n_events as stored, shape (n, 19, 4), plus the total number of events."""
    with h5py.File(path, "r") as handle:
        dataset = _particles(handle)
        return np.asarray(dataset[:n_events], dtype=np.float64), int(dataset.shape[0])


def summarize(particles: np.ndarray) -> dict:
    """Per object type: how many per event, how often present, typical leading pT."""
    has_type_column = particles.shape[2] >= 4
    present = particles[:, :, 3] > 0 if has_type_column else particles[:, :, 0] > 0
    summary = {"format_problems": []}
    for name, (first, last) in OBJECT_SLOTS.items():
        count = present[:, first:last].sum(axis=1)
        leading_pt = particles[:, first, 0]
        leading_pt = leading_pt[leading_pt > 0]
        summary[name] = {
            "mean_per_event": float(count.mean()),
            "fraction_with_any": float((count > 0).mean()),
            "count_histogram": np.bincount(count, minlength=last - first + 1).tolist(),
            "leading_pt_median": float(np.median(leading_pt)) if len(leading_pt) else None,
        }
        if has_type_column:  # every filled slot must carry the type of its group
            types = np.unique(particles[:, first:last, 3]).astype(int).tolist()
            if not set(types) <= {0, CLASS_INDEX[name]}:
                summary["format_problems"].append(
                    f"{name} slots {first}-{last - 1} contain particle types {types}, "
                    f"expected only 0 and {CLASS_INDEX[name]}")
    return summary


def print_event(event: np.ndarray) -> None:
    slot_names = {index: name for name, (first, last) in OBJECT_SLOTS.items()
                  for index in range(first, last)}
    print(f"  {'slot':>4}  {'object':<9}{'pT [GeV]':>10}{'eta':>8}{'phi':>8}{'type':>6}")
    for slot, row in enumerate(event):
        if not np.any(row != 0):
            continue  # empty slot (zero padding)
        kind = int(row[3]) if len(row) >= 4 else -1
        print(f"  {slot:>4}  {slot_names[slot]:<9}{row[0]:>10.2f}{row[1]:>8.3f}{row[2]:>8.3f}{kind:>6}")


def save_figure(particles: np.ndarray, summary: dict, title: str, path: Path) -> None:
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    for name in ("electron", "muon", "jet"):
        histogram = np.asarray(summary[name]["count_histogram"], dtype=float)
        left.plot(np.arange(len(histogram)), histogram / histogram.sum(), marker="o", label=name)
    left.set_xlabel("objects per event")
    left.set_ylabel("fraction of events")
    left.set_yscale("log")
    left.legend()

    bins = np.logspace(0, 3, 60)
    for name, (first, _) in OBJECT_SLOTS.items():
        pt = particles[:, first, 0]
        right.hist(pt[pt > 0], bins=bins, histtype="step", density=True,
                   label=f"{name} (leading)" if name != "MET" else "MET")
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xlabel("pT [GeV]")
    right.set_ylabel("probability density")
    right.legend(loc="lower left", fontsize=8)
    figure.suptitle(title)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Look inside a raw ADC2021 file.")
    parser.add_argument("file", type=Path)
    parser.add_argument("--events", type=int, default=200_000, help="how many events to read")
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = parser.parse_args(argv)
    if not args.file.exists():
        print(f"ERROR: {args.file} does not exist", file=sys.stderr)
        return 1

    print(f"file: {args.file}")
    for key, entry in describe_file(args.file)["datasets"].items():
        values = f"  values: {entry['values']}" if "values" in entry else ""
        print(f"  dataset {key!r}: shape {entry['shape']}, type {entry['dtype']}{values}")

    try:
        particles, total = read_raw(args.file, args.events)
    except (KeyError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"events in file: {total:,} (read the first {len(particles):,})")
    print(f"\nfirst event (empty slots not shown), stored under {PARTICLES_KEY!r}:")
    print_event(particles[0])

    summary = summarize(particles)
    print(f"\n{'object':<10}{'mean per event':>16}{'events with any':>18}{'median leading pT':>20}")
    for name in OBJECT_SLOTS:
        entry = summary[name]
        median = f"{entry['leading_pt_median']:.1f} GeV" if entry["leading_pt_median"] else "-"
        print(f"{name:<10}{entry['mean_per_event']:>16.3f}{100 * entry['fraction_with_any']:>17.1f}%{median:>20}")

    if summary["format_problems"]:
        print("\nFORMAT CHECK: unexpected content, the code may not read this file correctly:")
        for problem in summary["format_problems"]:
            print(f"  - {problem}")
    else:
        print("\nFORMAT CHECK: OK (slot layout matches the dataset paper)")

    figure_path = args.figures_dir / f"explore_{args.file.stem}.png"
    save_figure(particles, summary, f"{args.file.name} (first {len(particles):,} events)", figure_path)
    print(f"figure saved to {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
