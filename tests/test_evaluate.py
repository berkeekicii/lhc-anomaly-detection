import json

import numpy as np
import pytest
import torch

from lhcad.constants import SIGNALS
from lhcad.data import Scaler
from lhcad.evaluate import (anomaly_scores, evaluate_run, load_run, roc_auc, threshold_at_fpr,
                            tpr_at_fpr)
from lhcad.model import DenseAutoencoder
from lhcad.train import train_model


def test_threshold_lets_exactly_k_background_events_through():
    background = np.random.default_rng(0).permutation(1_000).astype(np.float32)  # scores 0..999
    threshold, k = threshold_at_fpr(background, fpr=0.01)
    assert k == 10 and threshold == 989
    assert int(np.sum(background > threshold)) == 10


def test_tpr_at_fpr():
    background = np.arange(1_000, dtype=np.float32)
    signal = np.array([3, 985, 995, 2_000], dtype=np.float32)
    assert tpr_at_fpr(background, signal, fpr=0.01) == 0.5


def test_threshold_needs_enough_background_events():
    with pytest.raises(ValueError, match="needs at least 100,000"):
        threshold_at_fpr(np.arange(1_000, dtype=np.float32), fpr=1e-5)


def test_roc_auc_limits():
    rng = np.random.default_rng(0)
    same = roc_auc(rng.normal(size=20_000), rng.normal(size=20_000))
    assert same == pytest.approx(0.5, abs=0.02)
    assert roc_auc(np.zeros(100), np.ones(100)) == 1.0


@pytest.fixture(scope="module")
def trained(processed_dir, tmp_path_factory):
    """A small trained model, saved the same way lhcad.train saves it."""
    run_dir = tmp_path_factory.mktemp("runs") / "test-run"
    run_dir.mkdir()
    train = np.load(processed_dir / "background_train.npy")
    val = np.load(processed_dir / "background_val.npy")
    scaler = Scaler.fit(train)
    torch.manual_seed(0)
    model = DenseAutoencoder()
    train_model(model, torch.from_numpy(scaler.transform(train)),
                torch.from_numpy(scaler.transform(val)), epochs=5, batch_size=256, quiet=True)
    torch.save({"config": model.config, "state_dict": model.state_dict()}, run_dir / "model.pt")
    scaler.save(run_dir / "scaler.npz")
    return {"run_dir": run_dir, "model": model, "scaler": scaler}


def test_saved_model_gives_the_same_scores_after_reloading(trained, processed_dir):
    events = np.load(processed_dir / "background_test.npy")[:500]
    before = anomaly_scores(trained["model"], trained["scaler"], events)
    model, scaler = load_run(trained["run_dir"])
    np.testing.assert_allclose(anomaly_scores(model, scaler, events, batch_size=128), before,
                               rtol=1e-5)


def test_evaluate_run_end_to_end(trained, processed_dir):
    run_dir = trained["run_dir"]
    metrics = evaluate_run(run_dir, processed_dir, quiet=True)
    assert json.loads((run_dir / "metrics.json").read_text()) == metrics
    assert (run_dir / "roc.png").stat().st_size > 0 and (run_dir / "scores.png").stat().st_size > 0
    assert metrics["background_test_events"] == 8_000
    assert set(metrics["signals"]) == set(SIGNALS)
    for name, entry in metrics["signals"].items():
        # the fake signals differ strongly from the fake background by construction
        assert entry["auc"] > 0.6, f"{name}: AUC {entry['auc']:.3f}"
        assert set(entry["tpr_at_fpr"]) == {"0.001"}, "8,000 test events only support FPR 1e-3"
        assert entry["background_above_threshold"]["0.001"] == 8
