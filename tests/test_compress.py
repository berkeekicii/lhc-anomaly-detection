import json

import numpy as np
import pytest
import torch
from torch import nn

from lhcad.compress import (compress_run, measure_latency, nonzero_weight_bytes, prune_model,
                            quantize_model, sparsity, weight_bytes)
from lhcad.data import Scaler
from lhcad.model import DenseAutoencoder, reconstruction_error
from lhcad.train import train_model


@pytest.fixture(scope="module")
def trained_run(processed_dir, tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("runs") / "r"
    run_dir.mkdir()
    train = np.load(processed_dir / "background_train.npy")
    val = np.load(processed_dir / "background_val.npy")
    scaler = Scaler.fit(train)
    torch.manual_seed(0)
    model = DenseAutoencoder()
    train_model(model, torch.from_numpy(scaler.transform(train)),
                torch.from_numpy(scaler.transform(val)), epochs=2, batch_size=256, quiet=True)
    torch.save({"config": model.config, "state_dict": model.state_dict()}, run_dir / "model.pt")
    scaler.save(run_dir / "scaler.npz")
    return run_dir


def test_pruning_zeroes_the_requested_share_and_keeps_the_largest_weights():
    torch.manual_seed(0)
    model = DenseAutoencoder()
    before = {id(m): m.weight.detach().clone() for m in model.modules() if isinstance(m, nn.Linear)}
    assert sparsity(model) == 0

    pruned = prune_model(model, 0.5)
    assert sparsity(pruned) == pytest.approx(0.5, abs=0.01)
    assert sparsity(model) == 0, "the original model is left alone"

    # every surviving weight is at least as large as every removed one
    kept, removed = [], []
    for original, new in zip(before.values(),
                             (m.weight.detach() for m in pruned.modules() if isinstance(m, nn.Linear))):
        mask = new != 0
        kept.append(original[mask].abs())
        removed.append(original[~mask].abs())
        assert torch.allclose(new[mask], original[mask]), "surviving weights are unchanged"
    assert torch.cat(kept).min() >= torch.cat(removed).max()


def test_pruning_survives_further_training_when_frozen(processed_dir):
    from lhcad.compress import freeze_pruned

    train = np.load(processed_dir / "background_train.npy")
    scaler = Scaler.fit(train)
    x = torch.from_numpy(scaler.transform(train))
    torch.manual_seed(0)
    pruned = prune_model(DenseAutoencoder(), 0.8)
    zeros_before = {id(m): (m.weight == 0).clone() for m in pruned.modules() if isinstance(m, nn.Linear)}

    handles = freeze_pruned(pruned)
    train_model(pruned, x, x[:500], epochs=2, batch_size=256, lr=1e-3, quiet=True)
    for handle in handles:
        handle.remove()

    assert sparsity(pruned) == pytest.approx(0.8, abs=0.01)
    for module in (m for m in pruned.modules() if isinstance(m, nn.Linear)):
        assert torch.equal((module.weight == 0), zeros_before[id(module)]), "same weights stay zero"


def test_byte_counts():
    torch.manual_seed(0)
    model = DenseAutoencoder(batch_norm=False)          # 4,924 float32 numbers
    assert weight_bytes(model) == 4924 * 4
    assert nonzero_weight_bytes(model) == weight_bytes(model)
    pruned = prune_model(model, 0.5)
    assert weight_bytes(pruned) == weight_bytes(model), "zeros still occupy their four bytes"
    assert nonzero_weight_bytes(pruned) < 0.6 * weight_bytes(model)


def test_quantized_model_is_smaller_and_still_scores_the_same_events():
    torch.manual_seed(0)
    model = DenseAutoencoder().eval()
    quantized = quantize_model(model)
    assert weight_bytes(quantized) < 0.5 * weight_bytes(model)
    x = torch.randn(64, 57)
    before, after = reconstruction_error(model, x), reconstruction_error(quantized, x)
    assert after.shape == before.shape
    assert torch.allclose(after, before, rtol=0.1, atol=0.05), "8-bit weights change the score only slightly"


def test_measure_latency_shapes_and_guard():
    model = DenseAutoencoder().eval()
    x = torch.randn(400, 57)
    result = measure_latency(model, x, batch_size=50, repeats=3, warmup=1)
    assert result["batch_size"] == 50 and result["timed_batches"] == 3
    assert result["median_us_per_event"] > 0
    assert result["fastest_us_per_event"] <= result["median_us_per_event"]
    with pytest.raises(ValueError, match="need at least"):
        measure_latency(model, x, batch_size=200, repeats=3, warmup=5)


def test_compress_run_end_to_end(trained_run, processed_dir):
    report = compress_run(trained_run, processed_dir, amounts=(0.5,), finetune_epochs=1,
                          batch_sizes=(1, 100), quiet=True)
    assert json.loads((trained_run / "compression.json").read_text()) == report
    assert set(report["variants"]) == {"float32", "pruned-50", "int8"}
    for name, entry in report["variants"].items():
        assert len(entry["signals"]) == 4
        assert [v["batch_size"] for v in entry["latency"]] == [1, 100]
        assert all(0.3 < s["auc"] < 1.0 for s in entry["signals"].values()), name
    assert report["variants"]["int8"]["weight_bytes"] < report["variants"]["float32"]["weight_bytes"]
    assert report["variants"]["pruned-50"]["zero_weights"] == pytest.approx(0.5, abs=0.01)
    assert report["variants"]["int8"]["zero_weights"] is None
