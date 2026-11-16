"""Draw the figures of a results folder written by lhcad.export.

Usage (from the project root):

    python -m lhcad.figures results/baseline

Writes loss.png, roc.png, scores.png and inputs.png into the same folder.
Needs only the small files in that folder, not the dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .constants import CHALLENGE_FPR, DATASETS
from .plots import plot_loss

LABELS = {name: info["label"] for name, info in DATASETS.items()}
LABELS["background"] = "SM background"


def plot_roc(curves: dict, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 5))
    for name, curve in curves["roc"].items():
        axis.plot(curve["fpr"], np.maximum(curve["tpr"], 1e-7), label=LABELS[name])
    diagonal = np.logspace(-7, 0, 50)
    axis.plot(diagonal, diagonal, "--", color="0.75", label="random guess")
    axis.axvline(CHALLENGE_FPR, color="red", linestyle="--", linewidth=1, label="FPR = 1e-5")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(1e-6, 1)
    axis.set_ylim(1e-5, 1.2)
    axis.set_xlabel("false positive rate (background kept)")
    axis.set_ylabel("true positive rate (signal kept)")
    axis.grid(True, which="major", alpha=0.3)
    axis.legend(loc="lower right", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_scores(curves: dict, path: Path) -> None:
    edges = np.asarray(curves["scores"]["edges"])
    widths = np.diff(edges)
    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    for name, counts in curves["scores"]["counts"].items():
        counts = np.asarray(counts, dtype=float)
        density = counts / (counts.sum() * widths)
        label = "SM background (test)" if name == "background" else LABELS[name]
        axis.stairs(density, edges, linewidth=1.3, label=label)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("anomaly score (reconstruction error)")
    axis.set_ylabel("probability density")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_inputs(curves: dict, path: Path) -> None:
    """What the events look like: objects per event and leading-object pT, per file."""
    inputs = curves["inputs"]
    edges = np.asarray(inputs["pt_edges"])
    objects = ("electron", "muon", "jet", "MET")
    figure, axes = plt.subplots(2, 4, figsize=(15, 6.5))
    for column, obj in enumerate(objects):
        for name, entry in inputs["files"].items():
            data = entry["objects"][obj]
            if obj != "MET":
                histogram = np.asarray(data["count_histogram"], dtype=float)
                axes[0, column].plot(np.arange(len(histogram)), histogram / histogram.sum(),
                                     marker="o", markersize=3, label=LABELS[name])
            counts = np.asarray(data["leading_pt_counts"], dtype=float)
            if counts.sum() > 0:
                axes[1, column].stairs(counts / (counts.sum() * np.diff(edges)), edges, label=LABELS[name])
        axes[0, column].set_title(f"{obj}s per event" if obj != "MET" else "MET (always present)")
        axes[0, column].set_yscale("log")
        axes[0, column].set_xlabel("objects per event")
        axes[1, column].set_title(f"leading {obj} pT" if obj != "MET" else "MET")
        axes[1, column].set_xscale("log")
        axes[1, column].set_yscale("log")
        axes[1, column].set_xlabel("pT [GeV]")
    axes[0, 3].axis("off")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_ylabel("fraction of events")
    axes[1, 0].set_ylabel("probability density")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def make_figures(folder) -> list[Path]:
    folder = Path(folder)
    curves = json.loads((folder / "curves.json").read_text())
    written = []
    plot_loss(json.loads((folder / "history.json").read_text()), folder / "loss.png")
    written.append(folder / "loss.png")
    plot_roc(curves, folder / "roc.png")
    written.append(folder / "roc.png")
    plot_scores(curves, folder / "scores.png")
    written.append(folder / "scores.png")
    if "inputs" in curves:
        plot_inputs(curves, folder / "inputs.png")
        written.append(folder / "inputs.png")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draw the figures of an exported results folder.")
    parser.add_argument("folder", type=Path)
    args = parser.parse_args(argv)
    if not (args.folder / "curves.json").exists():
        print(f"ERROR: {args.folder}/curves.json not found; run 'python -m lhcad.export' first",
              file=sys.stderr)
        return 1
    for path in make_figures(args.folder):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
