"""Make a trained model smaller and faster, and measure what that costs.

In a real trigger the decision has to be made in microseconds, so a model is
only useful if it is small and fast. This module takes a trained autoencoder
and produces three cheaper versions of it, then measures, for each one:

* how well it still separates the signals (AUC and TPR, as in lhcad.evaluate);
* how long one event takes on the CPU;
* how many bytes the weights take.

The variants:
    float32   the trained model, unchanged (the reference point)
    pruned    the smallest weights set to zero, then a few epochs of
              fine-tuning to let the remaining weights adjust
    int8      dynamic quantization: the linear layers store 8-bit integers
              instead of 32-bit floats and convert on the fly

Usage (from the project root):

    python -m lhcad.compress --run runs/long

Writes compression.json into the run directory.

A caution about the numbers: this measures PyTorch on a CPU, which is not what
a trigger runs on. The reference implementation (arXiv:2108.03986) reaches
80 ns on an FPGA. The point here is the comparison between the variants, not
the absolute times.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils import prune

from .constants import DATASETS, SIGNALS
from .data import Scaler
from .evaluate import FPR_POINTS, anomaly_scores, load_run, roc_auc, tpr_at_fpr
from .model import count_parameters, reconstruction_error
from .train import train_model


def linear_layers(model: nn.Module) -> list[nn.Linear]:
    return [m for m in model.modules() if isinstance(m, nn.Linear)]


def weight_bytes(model: nn.Module) -> int:
    """Bytes the weights occupy, counting the real storage of quantized ones."""
    total = 0
    for module in model.modules():
        for name in ("weight", "bias"):
            value = getattr(module, name, None)
            if callable(value):           # quantized layers expose weight() as a method
                value = value()
            if isinstance(value, torch.Tensor) and value.numel():
                total += value.numel() * value.element_size()
    return total


def sparsity(model: nn.Module) -> float:
    """Fraction of linear-layer weights that are exactly zero."""
    zeros = total = 0
    for layer in linear_layers(model):
        w = layer.weight
        zeros += int((w == 0).sum())
        total += w.numel()
    return zeros / max(total, 1)


def nonzero_weight_bytes(model: nn.Module) -> int:
    """Bytes the weights would take if the zeros were not stored at all.

    Pruning does not make a dense PyTorch tensor smaller: a zero still takes
    four bytes. It is still the number that matters for hardware that can skip
    the zeroed connections, such as the FPGA implementation of the reference
    paper, so both numbers are reported.
    """
    total = 0
    for module in model.modules():
        for name in ("weight", "bias"):
            value = getattr(module, name, None)
            if callable(value):
                value = value()
            if isinstance(value, torch.Tensor) and value.numel():
                total += int((value != 0).sum()) * value.element_size()
    return total


def prune_model(model: nn.Module, amount: float) -> nn.Module:
    """Set the given fraction of the smallest linear weights to zero, for good.

    Pruning is global: the threshold is chosen across all linear layers at
    once, so a layer whose weights matter more keeps more of them.
    """
    model = copy.deepcopy(model)
    parameters = [(layer, "weight") for layer in linear_layers(model)]
    prune.global_unstructured(parameters, pruning_method=prune.L1Unstructured, amount=amount)
    for layer, name in parameters:
        prune.remove(layer, name)        # make the zeros permanent
    return model


def freeze_pruned(model: nn.Module) -> list:
    """Keep the pruned weights at zero during fine-tuning.

    Returns the hook handles so the caller can remove them afterwards.
    """
    handles = []
    for layer in linear_layers(model):
        mask = (layer.weight != 0).to(layer.weight.dtype)
        handles.append(layer.weight.register_hook(lambda grad, mask=mask: grad * mask))
    return handles


def quantize_model(model: nn.Module) -> nn.Module:
    """8-bit dynamic quantization of the linear layers.

    PyTorch warns that this API is on its way out; if a future version drops
    it, the other variants still work and this one is reported as unavailable.
    """
    return torch.ao.quantization.quantize_dynamic(
        copy.deepcopy(model).eval(), {nn.Linear}, dtype=torch.qint8)


@torch.no_grad()
def measure_latency(model: nn.Module, x: torch.Tensor, batch_size: int,
                    repeats: int = 30, warmup: int = 5) -> dict:
    """Time one forward pass per event, in microseconds.

    Reports the median over `repeats` timed batches; the median rather than the
    mean because a single interruption by the operating system would dominate
    an average.
    """
    model.eval()
    batches = [x[i:i + batch_size] for i in range(0, min(len(x), batch_size * (warmup + repeats)),
                                                  batch_size)]
    batches = [b for b in batches if len(b) == batch_size]
    if len(batches) < warmup + 1:
        raise ValueError(f"need at least {batch_size * (warmup + 1)} events to time batches "
                         f"of {batch_size}, got {len(x)}")
    for batch in batches[:warmup]:
        reconstruction_error(model, batch)
    times = []
    for batch in batches[warmup:warmup + repeats]:
        started = time.perf_counter()
        reconstruction_error(model, batch)
        times.append((time.perf_counter() - started) / batch_size * 1e6)
    times = np.sort(np.asarray(times))
    return {"batch_size": batch_size, "timed_batches": len(times),
            "median_us_per_event": float(np.median(times)),
            "fastest_us_per_event": float(times[0])}


def evaluate_variant(model: nn.Module, scaler: Scaler, background: np.ndarray,
                     signals: dict[str, np.ndarray]) -> dict:
    """AUC and TPR of a model variant, exactly as lhcad.evaluate computes them."""
    background_scores = anomaly_scores(model, scaler, background)
    out = {}
    for name, events in signals.items():
        scores = anomaly_scores(model, scaler, events)
        entry = {"label": DATASETS[name]["label"], "auc": roc_auc(background_scores, scores),
                 "tpr_at_fpr": {}}
        for fpr in FPR_POINTS:
            if fpr * len(background_scores) >= 1:
                entry["tpr_at_fpr"][f"{fpr:g}"] = tpr_at_fpr(background_scores, scores, fpr)
        out[name] = entry
    return out


def compress_run(run_dir, processed_dir, amounts=(0.5, 0.8), finetune_epochs: int = 5,
                 max_eval_events: int = 0, batch_sizes=(1, 1000), seed: int = 0,
                 quiet: bool = False) -> dict:
    """Build the variants, evaluate and time them, and save compression.json."""
    say = (lambda *a, **k: None) if quiet else print
    run_dir, processed_dir = Path(run_dir), Path(processed_dir)
    torch.manual_seed(seed)
    model, scaler = load_run(run_dir)

    background = np.load(processed_dir / "background_test.npy", mmap_mode="r")
    if max_eval_events:
        background = background[:max_eval_events]
    signals = {name: np.load(processed_dir / f"signal_{name}.npy", mmap_mode="r")
               for name in SIGNALS if (processed_dir / f"signal_{name}.npy").exists()}
    timing_events = torch.from_numpy(scaler.transform(np.asarray(background[:200_000])))

    variants: list[tuple[str, nn.Module]] = [("float32", model)]
    for amount in amounts:
        name = f"pruned-{int(round(100 * amount))}"
        pruned = prune_model(model, amount)
        if finetune_epochs:
            train = np.load(processed_dir / "background_train.npy", mmap_mode="r")
            val = np.load(processed_dir / "background_val.npy", mmap_mode="r")
            handles = freeze_pruned(pruned)
            say(f"fine-tuning {name} for up to {finetune_epochs} epochs")
            train_model(pruned, torch.from_numpy(scaler.transform(train)),
                        torch.from_numpy(scaler.transform(val)),
                        epochs=finetune_epochs, patience=finetune_epochs, lr=1e-4,
                        seed=seed, quiet=quiet)
            for handle in handles:
                handle.remove()
        variants.append((name, pruned))
    try:
        variants.append(("int8", quantize_model(model)))
    except Exception as error:                      # the API is deprecated upstream
        say(f"int8 quantization unavailable in torch {torch.__version__}: {error}")
        report_note = f"int8 variant skipped: {error}"
    else:
        report_note = None

    float_size = weight_bytes(model)
    report = {"run": run_dir.name, "parameters": count_parameters(model),
              "background_test_events": int(len(background)), "variants": {}}
    for name, variant in variants:
        entry = {"weight_bytes": weight_bytes(variant),
                 "weight_bytes_relative": weight_bytes(variant) / float_size,
                 "nonzero_weight_bytes": nonzero_weight_bytes(variant),
                 "zero_weights": sparsity(variant) if name != "int8" else None,
                 "signals": evaluate_variant(variant, scaler, background, signals),
                 "latency": [measure_latency(variant, timing_events, b) for b in batch_sizes]}
        report["variants"][name] = entry
        say(f"{name}: {entry['weight_bytes']:,} bytes stored, "
            f"{entry['nonzero_weight_bytes']:,} non-zero, "
            + ", ".join(f"{v['median_us_per_event']:.1f} us/event at batch {v['batch_size']}"
                        for v in entry["latency"]))

    report["torch_threads"] = torch.get_num_threads()
    report["torch_version"] = torch.__version__
    if report_note:
        report["note"] = report_note
    (run_dir / "compression.json").write_text(json.dumps(report, indent=2))
    if not quiet:
        print_table(report)
    return report


def print_table(report: dict) -> None:
    signals = list(next(iter(report["variants"].values()))["signals"])
    batches = [v["batch_size"] for v in next(iter(report["variants"].values()))["latency"]]
    header = (f"{'variant':<12}{'stored':>9}{'non-zero':>10}"
              + "".join(f"{'us@' + str(b):>10}" for b in batches)
              + "".join(f"{DATASETS[s]['label']:>16}" for s in signals))
    print(f"\nrun {report['run']}, {report['background_test_events']:,} background test events, "
          f"{report['torch_threads']} CPU thread(s)")
    print(header)
    for name, entry in report["variants"].items():
        times = "".join(f"{v['median_us_per_event']:>10.1f}" for v in entry["latency"])
        aucs = "".join(f"{entry['signals'][s]['auc']:>16.3f}" for s in signals)
        print(f"{name:<12}{entry['weight_bytes']:>9,}{entry['nonzero_weight_bytes']:>10,}"
              f"{times}{aucs}")
    print("'stored' is the size on disk, 'non-zero' what it would be if zeros were dropped; "
          "pruning only changes the second.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Shrink a trained model and measure the cost.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--amounts", type=float, nargs="*", default=[0.5, 0.8],
                        help="fractions of weights to prune away (default: 0.5 0.8)")
    parser.add_argument("--finetune-epochs", type=int, default=5,
                        help="epochs of fine-tuning after pruning; 0 = none")
    parser.add_argument("--max-eval-events", type=int, default=0,
                        help="use at most this many background test events; 0 = all")
    parser.add_argument("--threads", type=int, default=1,
                        help="CPU threads for the timing (default 1, for comparable numbers)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if not (args.run / "model.pt").exists():
        print(f"ERROR: {args.run}/model.pt not found. Train a model first.", file=sys.stderr)
        return 1
    if args.threads:
        torch.set_num_threads(args.threads)
    try:
        compress_run(args.run, args.processed_dir, amounts=tuple(args.amounts),
                     finetune_epochs=args.finetune_epochs, max_eval_events=args.max_eval_events,
                     seed=args.seed)
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"\nsaved {args.run}/compression.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
