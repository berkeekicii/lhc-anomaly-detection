import numpy as np
import pytest
import torch

from lhcad.data import Scaler
from lhcad.model import DenseAutoencoder, count_parameters, reconstruction_error
from lhcad.train import mean_loss, train_model


@pytest.fixture(scope="module")
def tensors(processed_dir):
    train = np.load(processed_dir / "background_train.npy")
    val = np.load(processed_dir / "background_val.npy")
    scaler = Scaler.fit(train)
    return torch.from_numpy(scaler.transform(train)), torch.from_numpy(scaler.transform(val))


def test_default_model_has_the_layer_sizes_of_the_reference_paper():
    # 57->32->16->3->16->32->57 with biases: 1856 + 528 + 51 + 64 + 544 + 1881 = 4924
    assert count_parameters(DenseAutoencoder(batch_norm=False)) == 4924
    # batch normalisation adds a scale and a shift per unit of the five inner layers
    assert count_parameters(DenseAutoencoder()) == 4924 + 2 * (32 + 16 + 3 + 16 + 32)


def test_forward_and_error_shapes():
    model = DenseAutoencoder(hidden=(16,), latent_dim=3).eval()
    x = torch.randn(10, 57)
    assert model(x).shape == (10, 57)
    error = reconstruction_error(model, x)
    assert error.shape == (10,) and torch.all(error >= 0)


def test_training_reduces_the_validation_loss_and_keeps_the_best_weights(tensors):
    x_train, x_val = tensors
    torch.manual_seed(0)
    model = DenseAutoencoder()
    before = mean_loss(model, x_val)
    history = train_model(model, x_train, x_val, epochs=4, batch_size=256, quiet=True)
    assert len(history["train_loss"]) == len(history["val_loss"]) == 4
    assert history["best_val_loss"] < 0.8 * before
    assert history["best_val_loss"] == min(history["val_loss"])
    assert mean_loss(model, x_val) == pytest.approx(history["best_val_loss"], rel=1e-6)


def test_training_is_reproducible(tensors):
    x_train, x_val = tensors
    histories = []
    for _ in range(2):
        torch.manual_seed(0)
        history = train_model(DenseAutoencoder(), x_train, x_val, epochs=2, batch_size=256, quiet=True)
        histories.append(history["val_loss"])
    assert histories[0] == pytest.approx(histories[1], rel=1e-5)


def test_early_stopping(tensors):
    x_train, x_val = tensors
    torch.manual_seed(0)
    model = DenseAutoencoder(batch_norm=False)
    # learning rate 0: nothing changes, so only the first epoch counts as an improvement
    history = train_model(model, x_train, x_val, epochs=10, lr=0.0, patience=2, quiet=True)
    assert len(history["val_loss"]) == 3 and history["best_epoch"] == 1


def test_masked_error_ignores_empty_slots():
    torch.manual_seed(0)
    model = DenseAutoencoder(batch_norm=False, masked_loss=True).eval()
    x = torch.zeros(2, 57)
    x[0, 0:3] = torch.tensor([1.0, 0.0, 0.5])   # MET only
    x[1, 0:6] = torch.tensor([1.0, 0.0, 0.5, 2.0, 0.3, -1.0])  # MET and one electron
    squared = (model(x) - x) ** 2
    expected = torch.stack([squared[0, :3].mean(), squared[1, :6].mean()])
    assert torch.allclose(reconstruction_error(model, x), expected)
    plain = DenseAutoencoder(batch_norm=False)
    plain.load_state_dict(model.state_dict())
    assert torch.allclose(reconstruction_error(plain, x), squared.mean(dim=1))


def test_masked_loss_trains_and_is_restored_when_loading(tensors, tmp_path):
    from lhcad.evaluate import load_run

    x_train, x_val = tensors
    torch.manual_seed(0)
    model = DenseAutoencoder(masked_loss=True)
    before = mean_loss(model, x_val)
    history = train_model(model, x_train, x_val, epochs=2, batch_size=256, quiet=True)
    assert history["best_val_loss"] < before
    torch.save({"config": model.config, "state_dict": model.state_dict()}, tmp_path / "model.pt")
    Scaler.fit(np.zeros((4, 57), dtype=np.float32) + 1).save(tmp_path / "scaler.npz")
    loaded, _ = load_run(tmp_path)
    assert loaded.config["masked_loss"] is True
    assert torch.allclose(reconstruction_error(loaded, x_val[:100]),
                          reconstruction_error(model.eval(), x_val[:100]))
