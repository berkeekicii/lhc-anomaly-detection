"""Copy the results of a run into a small folder that can be kept in git.

Usage (from the project root, after training and evaluating):

    python -m lhcad.export --run runs/baseline
    python -m lhcad.figures results/baseline

The data and the full score arrays are far too large for git, but the numbers
a reader needs are small. This writes results/<run name>/ with:

    metrics.json        AUC and TPR values per signal
    history.json        loss after every epoch
    train_config.json   the options of the training run
    data_split.json     which events were used for training, validation, test
    curves.json         ROC curves and score histograms, reduced to a few hundred
                        points, plus object statistics of every input file
    model.pt            the trained weights (about 30 kB)
    scaler.npz          the numbers the inputs were divided by

--print-archive additionally prints the folder as one line of text, which is
how the results were copied out of a Google Colab session.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np

from .constants import OBJECT_SLOTS, SIGNALS
from .data import find_raw_files
from .evaluate import FPR_POINTS, anomaly_scores, load_run

ROC_POINTS = 241
HIST_BINS = 100


def roc_points(background: np.ndarray, signal: np.ndarray, n_points: int = ROC_POINTS) -> dict:
    """True-positive rate at false-positive rates spaced evenly on a log scale.

    The threshold for a false-positive rate f is chosen exactly as in
    lhcad.evaluate: only floor(f * N) background events lie above it.
    """
    n = len(background)
    descending = np.sort(background)[::-1]
    fpr = np.logspace(np.log10(1.0 / n), 0, n_points)
    k = np.clip(np.floor(fpr * n).astype(int), 1, n - 1)
    # the false-positive rates reported in metrics.json are always on the curve
    k = np.unique(np.concatenate([k, [int(np.floor(f * n)) for f in FPR_POINTS if f * n >= 1]]))
    thresholds = descending[k]
    signal_sorted = np.sort(signal)
    tpr = 1.0 - np.searchsorted(signal_sorted, thresholds, side="right") / len(signal)
    return {"fpr": (k / n).tolist(), "tpr": tpr.tolist()}


def score_histograms(scores: dict[str, np.ndarray], n_bins: int = HIST_BINS) -> dict:
    positive = np.concatenate([s[s > 0] for s in scores.values()])
    edges = np.logspace(np.log10(positive.min()), np.log10(positive.max()), n_bins + 1)
    return {"edges": edges.tolist(),
            "counts": {name: np.histogram(values, bins=edges)[0].tolist()
                       for name, values in scores.items()}}


def object_statistics(raw_dir: Path, n_events: int = 200_000) -> dict:
    """Objects per event and leading-object pT for every raw file (first n_events)."""
    from .explore import read_raw, summarize  # imported here: needs only h5py

    pt_edges = np.logspace(0, 3, 61)
    stats = {}
    for name, path in find_raw_files(raw_dir).items():
        particles, total = read_raw(path, n_events)
        summary = summarize(particles)
        entry = {"file": path.name, "events_in_file": total, "events_read": len(particles), "objects": {}}
        for obj, (first, _) in OBJECT_SLOTS.items():
            pt = particles[:, first, 0]
            entry["objects"][obj] = {
                "count_histogram": summary[obj]["count_histogram"],
                "fraction_with_any": summary[obj]["fraction_with_any"],
                "leading_pt_counts": np.histogram(pt[pt > 0], bins=pt_edges)[0].tolist(),
            }
        stats[name] = entry
    return {"pt_edges": pt_edges.tolist(), "files": stats}


def export_run(run_dir, processed_dir, out_dir, raw_dir=None) -> Path:
    run_dir, processed_dir, out_dir = Path(run_dir), Path(processed_dir), Path(out_dir)
    if not (run_dir / "metrics.json").exists():
        raise FileNotFoundError(f"{run_dir}/metrics.json not found; run 'python -m lhcad.evaluate' first")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("metrics.json", "history.json", "train_config.json", "model.pt", "scaler.npz"):
        shutil.copy2(run_dir / name, out_dir / name)
    if (processed_dir / "meta.json").exists():
        shutil.copy2(processed_dir / "meta.json", out_dir / "data_split.json")

    model, scaler = load_run(run_dir)
    scores = {"background": anomaly_scores(model, scaler,
                                           np.load(processed_dir / "background_test.npy", mmap_mode="r"))}
    for name in SIGNALS:
        path = processed_dir / f"signal_{name}.npy"
        if path.exists():
            scores[name] = anomaly_scores(model, scaler, np.load(path, mmap_mode="r"))
    curves = {"roc": {name: roc_points(scores["background"], values)
                      for name, values in scores.items() if name != "background"},
              "scores": score_histograms(scores)}
    if raw_dir is not None and Path(raw_dir).exists():
        curves["inputs"] = object_statistics(Path(raw_dir))
    (out_dir / "curves.json").write_text(json.dumps(curves))
    return out_dir


def archive_text(folder: Path) -> str:
    """The folder as a gzip-compressed tar archive, encoded as one line of base64 text."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for path in sorted(Path(folder).iterdir()):
            tar.add(path, arcname=path.name)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Copy a run's results into results/<run>/.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--print-archive", action="store_true")
    args = parser.parse_args(argv)
    try:
        out = export_run(args.run, args.processed_dir, args.results_dir / args.run.name, args.raw_dir)
    except FileNotFoundError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"results written to {out}/")
    if args.print_archive:
        print("@@ARCHIVE@@" + archive_text(out) + "@@END@@")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
