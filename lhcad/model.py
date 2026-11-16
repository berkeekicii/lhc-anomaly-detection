"""The autoencoder.

An autoencoder is a network that is trained to copy its input to its output
through a narrow middle layer (the "latent space"). It only sees ordinary
(background) events during training, so it learns to compress those well.
An unusual event is compressed badly, and the size of the reconstruction error
is used as the anomaly score.

Default layer sizes follow the DNN autoencoder of Govorkova et al.
(arXiv:2108.03986): 57 -> 32 -> 16 -> 3 -> 16 -> 32 -> 57.
"""

from __future__ import annotations

import torch
from torch import nn

from .constants import INPUT_DIM


class DenseAutoencoder(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, hidden: tuple[int, ...] = (32, 16),
                 latent_dim: int = 3, batch_norm: bool = True, negative_slope: float = 0.3,
                 masked_loss: bool = False):
        super().__init__()
        # masked_loss: see reconstruction_error. It is stored with the model so
        # that training and evaluation always use the same anomaly score.
        self.config = {"input_dim": input_dim, "hidden": list(hidden), "latent_dim": latent_dim,
                       "batch_norm": batch_norm, "negative_slope": negative_slope,
                       "masked_loss": masked_loss}

        def block(n_in: int, n_out: int) -> list[nn.Module]:
            layers: list[nn.Module] = [nn.Linear(n_in, n_out)]
            if batch_norm:
                layers.append(nn.BatchNorm1d(n_out))
            layers.append(nn.LeakyReLU(negative_slope))
            return layers

        encoder: list[nn.Module] = []
        width = input_dim
        for size in (*hidden, latent_dim):
            encoder += block(width, size)
            width = size
        self.encoder = nn.Sequential(*encoder)

        decoder: list[nn.Module] = []
        for size in reversed(hidden):
            decoder += block(width, size)
            width = size
        decoder.append(nn.Linear(width, input_dim))  # no activation on the output
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def count_parameters(model: nn.Module) -> int:
    """Number of trainable numbers in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def reconstruction_error(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Mean squared difference between input and output, one value per event.

    With masked_loss, only the object slots that are filled in the input count:
    an event with one electron and two jets is judged on those objects and the
    MET, not on how well the network outputs zeros for the 15 empty slots. This
    follows Govorkova et al. (arXiv:2108.03986). It needs inputs in which an
    empty slot is exactly zero, i.e. a scaling that does not shift the values.
    """
    squared = (model(x) - x) ** 2
    if not getattr(model, "config", {}).get("masked_loss"):
        return squared.mean(dim=1)
    n_slots = x.shape[1] // 3
    present = (x.reshape(-1, n_slots, 3) != 0).any(dim=2)          # (events, slots)
    mask = present.repeat_interleave(3, dim=1).to(squared.dtype)   # (events, 57)
    return (squared * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
