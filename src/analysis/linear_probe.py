"""Frozen-feature linear probes.

Features are global-average-pooled block outputs on un-augmented, normalised images: the last
block's (``extract_features``) or every block's (``extract_layer_features``). They are standardised
(by default with the probe's own training set; for labelled-subset probes, with statistics of all
unlabelled training images), so the probe is insensitive to each method's activation scale. A
multinomial logistic regression is fit by full-batch L-BFGS with an L2 penalty.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.augmentations import CIFAR10_MEAN, CIFAR10_STD
from src.data.cifar import NUM_CLASSES


def normalize_images(images: torch.Tensor) -> torch.Tensor:
    """uint8 ``(N, 3, H, W)`` to CIFAR-10-normalised float."""
    mean = torch.tensor(CIFAR10_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=images.device).view(1, 3, 1, 1)
    return (images.float() / 255 - mean) / std


@torch.no_grad()
def extract_features(
    encoder: nn.Module, images: torch.Tensor, device: torch.device, batch_size: int = 1000
) -> torch.Tensor:
    """Pooled last-block features ``(N, 512)`` of uint8 images, computed in batches."""
    was_training = encoder.training
    encoder.eval()
    features = []
    for start in range(0, images.shape[0], batch_size):
        x = normalize_images(images[start : start + batch_size].to(device))
        features.append(encoder(x)[0])
    encoder.train(was_training)
    return torch.cat(features)


@torch.no_grad()
def extract_layer_features(
    encoder: nn.Module, images: torch.Tensor, device: torch.device, batch_size: int = 1000
) -> list[torch.Tensor]:
    """Pooled outputs of every block, ``[(N, C_1), ..., (N, C_8)]``; needs a layer-local encoder."""
    was_training = encoder.training
    encoder.eval()
    chunks = []
    for start in range(0, images.shape[0], batch_size):
        x = normalize_images(images[start : start + batch_size].to(device))
        chunks.append(encoder(x)[2])
    encoder.train(was_training)
    return [torch.cat([chunk[i] for chunk in chunks]) for i in range(len(chunks[0]))]


def linear_probe_accuracy(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    l2: float = 1e-4,
    max_iter: int = 200,
    stats: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> float:
    """Test accuracy of an L2-regularised logistic regression on standardised features.

    Args:
        stats: ``(mean, std)`` for standardisation; default: computed from ``train_x``.
    """
    if stats is None:
        stats = (train_x.mean(dim=0), train_x.std(dim=0))
    mean, std = stats[0], stats[1].clamp_min(1e-6)
    train_x, test_x = (train_x - mean) / std, (test_x - mean) / std
    train_y, test_y = train_y.to(train_x.device), test_y.to(test_x.device)
    classifier = nn.Linear(train_x.shape[1], NUM_CLASSES).to(train_x.device)
    optimizer = torch.optim.LBFGS(
        classifier.parameters(), max_iter=max_iter, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = F.cross_entropy(classifier(train_x), train_y) + l2 * classifier.weight.pow(2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        return (classifier(test_x).argmax(dim=1) == test_y).float().mean().item()
