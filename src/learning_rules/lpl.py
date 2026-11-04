"""LPL loss, re-implemented from upstream ``LPL.loss_step`` and ``training_step``.

Upstream: ``vendor/lpl_zenke/models/module.py``.

For each layer i, with ``z1[i]`` and ``z2[i]`` the pooled activations of the two members of each
pair:

- pull = MSE(z1, z2), the predictive term. Like upstream there is no stop-gradient: both members
  receive gradient.
- push = (H(z1) + H(z2)) / 2 with H(z) = -mean_j log(Var_b z_j + 1e-4), the Hebbian term. The
  batch mean is detached; the variance uses the unbiased (B - 1) normaliser.
- decorr = (D(z1) + D(z2)) / 2 with D(z) the mean squared off-diagonal covariance.

total = pull_coeff * sum_i pull_i + push_coeff * sum_i push_i + decorr_coeff * sum_i decorr_i.

Layer-local training comes from the encoder (``src/models/vgg.py``), which detaches each block's
input. Upstream's ``--shuffle_positives`` permutes z2 across the batch; since the encoder treats
samples independently and push/decorr ignore sample order, shuffling the second frames before the
encoder is identical (``src/data/sequences.frame_pairs(shuffle_positives=True)``).
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

HEBBIAN_EPS = 1e-4


@dataclass(frozen=True)
class LPLCoefficients:
    """Loss weights; defaults are upstream's."""

    pull: float = 1.0
    push: float = 1.0
    decorr: float = 10.0


def hebbian_loss(z: torch.Tensor) -> torch.Tensor:
    """Negative mean log variance of each unit over the batch (upstream ``hebbian_loss``)."""
    center = z.mean(dim=0).detach()
    variance = ((z - center) ** 2).sum(dim=0) / (z.shape[0] - 1)
    return -torch.log(variance + HEBBIAN_EPS).mean()


def decorrelation_loss(z: torch.Tensor) -> torch.Tensor:
    """Mean squared off-diagonal covariance between units (upstream ``decorr_loss``)."""
    centered = z - z.mean(dim=0).detach()
    cov = (centered.T @ centered).fill_diagonal_(0) / (z.shape[0] - 1)
    num_units = cov.shape[0]
    return (cov**2).sum() / (num_units**2 - num_units)


def lpl_layer_losses(
    z1: list[torch.Tensor], z2: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-layer pull, push and decorrelation losses, each of shape ``(num_layers,)``."""
    pull = torch.stack([F.mse_loss(a, b) for a, b in zip(z1, z2, strict=True)])
    push = torch.stack(
        [0.5 * (hebbian_loss(a) + hebbian_loss(b)) for a, b in zip(z1, z2, strict=True)]
    )
    decorr = torch.stack(
        [0.5 * (decorrelation_loss(a) + decorrelation_loss(b)) for a, b in zip(z1, z2, strict=True)]
    )
    return pull, push, decorr


def lpl_loss(
    z1: list[torch.Tensor],
    z2: list[torch.Tensor],
    coefficients: LPLCoefficients | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Total LPL loss and the per-layer terms (detached) for logging."""
    coefficients = coefficients or LPLCoefficients()
    pull, push, decorr = lpl_layer_losses(z1, z2)
    total = (
        coefficients.pull * pull.sum()
        + coefficients.push * push.sum()
        + coefficients.decorr * decorr.sum()
    )
    terms = {"pull": pull.detach(), "push": push.detach(), "decorr": decorr.detach()}
    return total, terms
