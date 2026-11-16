"""Train the autoencoder on background events.

Usage (from the project root):

    python -m lhcad.train                       # default model, saved to runs/<date-time>/
    python -m lhcad.train --name latent8 --latent-dim 8
    python -m lhcad.train --name raw-inputs --scaling none

A run directory holds everything needed to reuse and to reproduce the model:

    model.pt            weights and layer sizes
    scaler.npz          the numbers the inputs were shifted and divided by
    history.json        loss after every epoch
    train_config.json   the options this run was started with
    loss.png            training curve
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .data import SCALINGS, Scaler
from .model import DenseAutoencoder, count_parameters, reconstruction_error
from .plots import plot_loss


@torch.no_grad()
def mean_loss(model: torch.nn.Module, x: torch.Tensor, batch_size: int = 65536) -> float:
    """Average reconstruction error over all events in x (no training)."""
    model.eval()
    total = 0.0
    for start in range(0, len(x), batch_size):
        total += reconstruction_error(model, x[start:start + batch_size]).sum().item()
    return total / len(x)


def train_model(model: torch.nn.Module, x_train: torch.Tensor, x_val: torch.Tensor,
                epochs: int = 20, batch_size: int = 1024, lr: float = 1e-3,
                patience: int = 5, seed: int = 0, quiet: bool = False) -> dict:
    """Train with Adam and early stopping. The model ends up with its best weights.

    Early stopping: after every epoch the loss is measured on validation events
    the model has not trained on. If it has not improved for `patience` epochs
    in a row, training stops and the best weights seen so far are restored.
    """
    shuffle = torch.Generator().manual_seed(seed)
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": [], "seconds": []}
    best_val, best_epoch, best_state, epochs_without_gain = float("inf"), 0, None, 0

    for epoch in range(1, epochs + 1):
        started = time.time()
        model.train()
        order = torch.randperm(len(x_train), generator=shuffle).to(device)
        running, seen = 0.0, 0
        for start in range(0, len(order), batch_size):
            batch = x_train[order[start:start + batch_size]]
            if len(batch) < 2:  # batch normalisation cannot work with a single event
                continue
            loss = reconstruction_error(model, batch).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * len(batch)
            seen += len(batch)

        train_loss = running / max(seen, 1)
        val_loss = mean_loss(model, x_val)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["seconds"].append(time.time() - started)

        improved = val_loss < best_val
        if improved:
            best_val, best_epoch, epochs_without_gain = val_loss, epoch, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            epochs_without_gain += 1
        if not quiet:
            print(f"epoch {epoch:3d}/{epochs}  train {train_loss:.5f}  val {val_loss:.5f}  "
                  f"{history['seconds'][-1]:5.1f} s{'  *' if improved else ''}")
        if epochs_without_gain >= patience:
            if not quiet:
                print(f"no improvement for {patience} epochs, stopping")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val
    return history


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the autoencoder on background events.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--name", default=None, help="name of the run directory (default: date-time)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of the Adam optimizer")
    parser.add_argument("--patience", type=int, default=5, help="early-stopping patience in epochs")
    parser.add_argument("--hidden", type=int, nargs="+", default=[32, 16],
                        help="hidden layer sizes of the encoder (the decoder mirrors them)")
    parser.add_argument("--latent-dim", type=int, default=3)
    parser.add_argument("--no-batch-norm", action="store_true")
    parser.add_argument("--scaling", choices=SCALINGS, default="quantity",
                        help="how inputs are scaled, see lhcad/data.py (default: quantity)")
    parser.add_argument("--masked-loss", action="store_true",
                        help="ignore empty object slots in the loss and the anomaly score")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="where to train; auto = a GPU if there is one")
    parser.add_argument("--max-train-events", type=int, default=0, help="0 = use all")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    train_path = args.processed_dir / "background_train.npy"
    val_path = args.processed_dir / "background_val.npy"
    if not train_path.exists() or not val_path.exists():
        print(f"ERROR: {train_path} not found. Run 'python -m lhcad.prepare' first.", file=sys.stderr)
        return 1

    if args.masked_loss and args.scaling == "feature":
        print("ERROR: --masked-loss needs empty slots to stay zero; use --scaling quantity or none.",
              file=sys.stderr)
        return 1
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else \
        ("cpu" if args.device == "auto" else args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train = np.load(train_path, mmap_mode="r")
    val = np.load(val_path, mmap_mode="r")
    if args.max_train_events:
        train = train[:args.max_train_events]
    scaler = Scaler.fit(train, mode=args.scaling)
    x_train = torch.from_numpy(scaler.transform(train)).to(device)
    x_val = torch.from_numpy(scaler.transform(val)).to(device)

    model = DenseAutoencoder(input_dim=train.shape[1], hidden=tuple(args.hidden),
                             latent_dim=args.latent_dim, batch_norm=not args.no_batch_norm,
                             masked_loss=args.masked_loss).to(device)
    print(f"training on {len(x_train):,} events, validating on {len(x_val):,}, "
          f"input scaling: {args.scaling}, masked loss: {args.masked_loss}, device: {device}")
    print(f"model: {train.shape[1]} -> {' -> '.join(map(str, args.hidden))} -> {args.latent_dim} "
          f"-> ... -> {train.shape[1]}, {count_parameters(model):,} parameters")

    history = train_model(model, x_train, x_val, epochs=args.epochs, batch_size=args.batch_size,
                          lr=args.lr, patience=args.patience, seed=args.seed)

    run_dir = args.runs_dir / (args.name or datetime.now().strftime("run-%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    model.cpu()
    torch.save({"config": model.config, "state_dict": model.state_dict()}, run_dir / "model.pt")
    scaler.save(run_dir / "scaler.npz")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    config = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    config.update({"train_events": len(x_train), "val_events": len(x_val),
                   "parameters": count_parameters(model), "torch_version": torch.__version__})
    (run_dir / "train_config.json").write_text(json.dumps(config, indent=2))
    plot_loss(history, run_dir / "loss.png")

    print(f"best validation loss {history['best_val_loss']:.5f} at epoch {history['best_epoch']}")
    print(f"saved to {run_dir}/  -> next: python -m lhcad.evaluate --run {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
