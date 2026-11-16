"""Turn the raw HDF5 files into ready-to-train arrays.

Usage (from the project root):

    python -m lhcad.prepare
    python -m lhcad.prepare --max-events 1000000     # less memory, less precise test

Reads data/raw/*.h5, writes data/processed/*.npy :

    background_train.npy, background_val.npy, background_test.npy
    signal_<name>.npy (one per signal file found)
    meta.json (what was done, so results can be reproduced)

Memory: one event is 57 float32 numbers = 228 bytes, so 4 million events
need about 0.9 GB of RAM while this script runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .constants import BACKGROUND, DATASETS, SIGNALS
from .data import count_events, find_raw_files, load_events, split_background


def prepare(raw_dir: Path, out_dir: Path, max_events: int | None = 4_000_000,
            fractions=(0.5, 0.1, 0.4), seed: int = 0, quiet: bool = False) -> dict:
    """Create the processed arrays and return the metadata that was written."""
    say = (lambda *a, **k: None) if quiet else print
    files = find_raw_files(raw_dir)
    if BACKGROUND not in files:
        raise FileNotFoundError(
            f"no background file found in {raw_dir}/ . Run 'python -m lhcad.download' first, "
            f"or download it by hand from https://zenodo.org/records/{DATASETS[BACKGROUND]['record']}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_in_file = count_events(files[BACKGROUND])
    say(f"background: {files[BACKGROUND].name} holds {n_in_file:,} events")
    events = load_events(files[BACKGROUND], max_events=max_events, seed=seed)
    train, val, test = split_background(events, fractions=fractions, seed=seed)
    for part, array in (("train", train), ("val", val), ("test", test)):
        np.save(out_dir / f"background_{part}.npy", array)
        say(f"  background_{part}.npy  {len(array):>10,} events")
    meta = {
        "seed": seed,
        "fractions": list(fractions),
        "background": {"file": files[BACKGROUND].name, "events_in_file": n_in_file,
                       "events_used": int(len(events)), "train": int(len(train)),
                       "val": int(len(val)), "test": int(len(test))},
        "signals": {},
    }
    del events, train, val, test

    for name in SIGNALS:
        if name not in files:
            say(f"signal {name}: file not found in {raw_dir}/ (skipped)")
            continue
        signal = load_events(files[name])
        np.save(out_dir / f"signal_{name}.npy", signal)
        meta["signals"][name] = {"file": files[name].name, "events": int(len(signal))}
        say(f"  signal_{name}.npy  {len(signal):>10,} events  ({DATASETS[name]['label']})")

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare ADC2021 arrays for training.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--max-events", type=int, default=4_000_000,
                        help="use at most this many background events; 0 = all (default: 4,000,000)")
    parser.add_argument("--fractions", type=float, nargs=3, default=(0.5, 0.1, 0.4),
                        metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        prepare(args.raw_dir, args.out_dir, max_events=args.max_events or None,
                fractions=tuple(args.fractions), seed=args.seed)
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Done. Arrays are in {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
