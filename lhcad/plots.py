"""Figures. Everything is written to PNG files, no window is opened."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # works without a display

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from .constants import CHALLENGE_FPR, DATASETS


def plot_loss(history: dict, path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(epochs, history["train_loss"], label="training")
    axis.plot(epochs, history["val_loss"], label="validation")
    if history.get("best_epoch"):
        axis.axvline(history["best_epoch"], color="0.6", linestyle=":", label="best epoch")
    axis.set_xlabel("epoch")
    axis.set_ylabel("reconstruction error (MSE)")
    axis.set_yscale("log")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_roc(scores: dict[str, np.ndarray], path) -> None:
    """ROC curve of every signal against the background, on log axes."""
    background = scores["background"]
    figure, axis = plt.subplots(figsize=(6.5, 5))
    for name, signal in scores.items():
        if name == "background":
            continue
        labels = np.concatenate([np.zeros(len(background)), np.ones(len(signal))])
        fpr, tpr, _ = roc_curve(labels, np.concatenate([background, signal]))
        keep = fpr > 0
        axis.plot(fpr[keep], tpr[keep], label=DATASETS[name]["label"])
    diagonal = np.logspace(-6, 0, 50)
    axis.plot(diagonal, diagonal, "--", color="0.75", label="random guess")
    axis.axvline(CHALLENGE_FPR, color="red", linestyle="--", linewidth=1, label="FPR = 1e-5")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(1e-6, 1)
    axis.set_ylim(1e-6, 1.2)
    axis.set_xlabel("false positive rate (background kept)")
    axis.set_ylabel("true positive rate (signal kept)")
    axis.grid(True, which="major", alpha=0.3)
    axis.legend(loc="lower right", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_scores(scores: dict[str, np.ndarray], path) -> None:
    """Distribution of the anomaly score for background and for every signal."""
    positive = np.concatenate([s[s > 0] for s in scores.values()])
    bins = np.logspace(np.log10(positive.min()), np.log10(positive.max()), 100)
    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    for name, values in scores.items():
        label = "SM background (test)" if name == "background" else DATASETS[name]["label"]
        axis.hist(values, bins=bins, histtype="step", density=True, linewidth=1.3, label=label)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("anomaly score (reconstruction error)")
    axis.set_ylabel("probability density")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
