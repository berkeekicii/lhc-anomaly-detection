"""Measure how well a trained model separates signal events from background.

Usage (from the project root):

    python -m lhcad.evaluate                    # the most recent run in runs/
    python -m lhcad.evaluate --run runs/latent8

Every event gets an anomaly score (its reconstruction error). Two numbers
summarise the separation for each signal:

* AUC: the probability that a random signal event scores higher than a random
  background event (0.5 = no separation, 1.0 = perfect).
* TPR at a fixed FPR: put the threshold so that only a fraction FPR of the
  background events lie above it, then count the fraction of signal events
  above it. The challenge metric uses FPR = 1e-5: a trigger has to discard
  almost every collision, so only a tiny background rate is affordable.

Writes metrics.json, roc.png and scores.png into the run directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .constants import CHALLENGE_FPR, DATASETS, SIGNALS
from .data import Scaler
from .model import DenseAutoencoder, count_parameters, reconstruction_error
from .plots import plot_roc, plot_scores

FPR_POINTS = (1e-3, 1e-4, CHALLENGE_FPR)


def load_run(run_dir) -> tuple[DenseAutoencoder, Scaler]:
    """Rebuild a trained model and its input scaler from a run directory."""
    run_dir = Path(run_dir)
    stored = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
    config = stored["config"]
    model = DenseAutoencoder(input_dim=config["input_dim"], hidden=tuple(config["hidden"]),
                             latent_dim=config["latent_dim"], batch_norm=config["batch_norm"],
                             negative_slope=config["negative_slope"],
                             masked_loss=config.get("masked_loss", False))
    model.load_state_dict(stored["state_dict"])
    model.eval()
    return model, Scaler.load(run_dir / "scaler.npz")


@torch.no_grad()
def anomaly_scores(model: torch.nn.Module, scaler: Scaler, events: np.ndarray,
                   batch_size: int = 65536) -> np.ndarray:
    """Reconstruction error of every event, computed block by block."""
    model.eval()
    scores = np.empty(len(events), dtype=np.float32)
    for start in range(0, len(events), batch_size):
        block = torch.from_numpy(scaler.transform(events[start:start + batch_size]))
        scores[start:start + len(block)] = reconstruction_error(model, block).numpy()
    return scores


def threshold_at_fpr(background_scores: np.ndarray, fpr: float) -> tuple[float, int]:
    """Score threshold that lets through a fraction fpr of the background.

    Returns (threshold, k) where k = floor(fpr * N) is the number of background
    events allowed above the threshold. The threshold is the (k+1)-th largest
    background score, so exactly k background events are strictly above it
    (unless several events share that score).
    """
    n = len(background_scores)
    k = int(np.floor(fpr * n))
    if k < 1:
        raise ValueError(f"FPR {fpr:g} needs at least {int(np.ceil(1 / fpr)):,} background "
                         f"events, got {n:,}")
    position = n - k - 1
    return float(np.partition(background_scores, position)[position]), k


def tpr_at_fpr(background_scores: np.ndarray, signal_scores: np.ndarray, fpr: float) -> float:
    """Fraction of signal events above the threshold defined by the background."""
    threshold, _ = threshold_at_fpr(background_scores, fpr)
    return float(np.mean(signal_scores > threshold))


def roc_auc(background_scores: np.ndarray, signal_scores: np.ndarray) -> float:
    labels = np.concatenate([np.zeros(len(background_scores)), np.ones(len(signal_scores))])
    scores = np.concatenate([background_scores, signal_scores])
    return float(roc_auc_score(labels, scores))


def evaluate_run(run_dir, processed_dir, quiet: bool = False) -> dict:
    """Score the background test set and every signal; return and save the metrics."""
    run_dir, processed_dir = Path(run_dir), Path(processed_dir)
    model, scaler = load_run(run_dir)
    background = anomaly_scores(model, scaler,
                                np.load(processed_dir / "background_test.npy", mmap_mode="r"))
    metrics = {"run": run_dir.name, "scaling": scaler.mode, "parameters": count_parameters(model),
               "background_test_events": int(len(background)), "signals": {}}
    scores = {"background": background}
    for name in SIGNALS:
        path = processed_dir / f"signal_{name}.npy"
        if not path.exists():
            continue
        signal = anomaly_scores(model, scaler, np.load(path, mmap_mode="r"))
        scores[name] = signal
        entry = {"label": DATASETS[name]["label"], "events": int(len(signal)),
                 "auc": roc_auc(background, signal), "tpr_at_fpr": {}, "background_above_threshold": {}}
        for fpr in FPR_POINTS:
            try:
                _, k = threshold_at_fpr(background, fpr)
            except ValueError:
                continue  # not enough background events for this FPR
            entry["tpr_at_fpr"][f"{fpr:g}"] = tpr_at_fpr(background, signal, fpr)
            entry["background_above_threshold"][f"{fpr:g}"] = k
        metrics["signals"][name] = entry

    if not metrics["signals"]:
        raise FileNotFoundError(f"no signal_*.npy files in {processed_dir}/")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_roc(scores, run_dir / "roc.png")
    plot_scores(scores, run_dir / "scores.png")
    if not quiet:
        print_table(metrics)
    return metrics


def print_table(metrics: dict) -> None:
    fprs = [f"{fpr:g}" for fpr in FPR_POINTS]
    print(f"run {metrics['run']}: {metrics['parameters']:,} parameters, scaling {metrics['scaling']}, "
          f"{metrics['background_test_events']:,} background test events")
    print(f"{'signal':<18}{'events':>10}{'AUC':>9}" + "".join(f"{'TPR@' + f:>12}" for f in fprs))
    for entry in metrics["signals"].values():
        cells = "".join(f"{100 * entry['tpr_at_fpr'][f]:>11.3f}%" if f in entry["tpr_at_fpr"]
                        else f"{'n/a':>12}" for f in fprs)
        print(f"{entry['label']:<18}{entry['events']:>10,}{entry['auc']:>9.4f}{cells}")
    first = next(iter(metrics["signals"].values()))
    counts = ", ".join(f"{f}: {k}" for f, k in first["background_above_threshold"].items())
    print(f"background events above each threshold ({counts}); "
          "a TPR that rests on fewer than about 100 of them is a rough estimate")


def latest_run(runs_dir: Path) -> Path | None:
    runs = [p for p in Path(runs_dir).glob("*") if (p / "model.pt").exists()]
    return max(runs, key=lambda p: (p / "model.pt").stat().st_mtime) if runs else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained autoencoder.")
    parser.add_argument("--run", type=Path, default=None, help="run directory (default: newest in runs/)")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args(argv)

    run_dir = args.run or latest_run(args.runs_dir)
    if run_dir is None or not (run_dir / "model.pt").exists():
        print("ERROR: no trained run found. Run 'python -m lhcad.train' first.", file=sys.stderr)
        return 1
    try:
        evaluate_run(run_dir, args.processed_dir)
    except FileNotFoundError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"saved metrics.json, roc.png and scores.png to {run_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
