"""CIFAR-10 loading.

Reads the extracted python batches (``cifar-10-batches-py``) into uint8 tensors, and samples the
labelled subsets used for fine-tuning and linear probes.
"""

import pickle
from pathlib import Path

import numpy as np
import torch

from src.utils.paths import data_root

NUM_CLASSES = 10
TRAIN_FILES = tuple(f"data_batch_{i}" for i in range(1, 6))
TEST_FILES = ("test_batch",)


def cifar10_dir() -> Path:
    """Location of the extracted CIFAR-10 python batches."""
    return data_root() / "cifar10" / "cifar-10-batches-py"


def load_cifar10(
    split: str = "train", root: Path | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a CIFAR-10 split.

    Args:
        split: ``"train"`` (50,000 images) or ``"test"`` (10,000 images).
        root: Directory with the python batches. Defaults to ``cifar10_dir()``.

    Returns:
        Images as uint8 ``(N, 3, 32, 32)`` and labels as int64 ``(N,)``.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    root = Path(root) if root is not None else cifar10_dir()
    if not root.is_dir():
        raise FileNotFoundError(
            f"CIFAR-10 not found at {root}. Download and extract cifar-10-python.tar.gz there "
            "(see README), or set PREWIRE_DATA."
        )

    images, labels = [], []
    for name in TRAIN_FILES if split == "train" else TEST_FILES:
        with open(root / name, "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        images.append(batch[b"data"])
        labels.extend(batch[b"labels"])

    images = np.concatenate(images).reshape(-1, 3, 32, 32)
    return torch.from_numpy(images), torch.tensor(labels, dtype=torch.long)


def labeled_subset(labels: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    """Indices of a class-balanced labelled subset of size ``n``.

    Each class's training indices are shuffled once with ``seed`` and the first ``n / 10`` are
    taken, so for a given seed the subsets are nested (the 100-label set contains the 10-label
    set) and deterministic.

    Args:
        labels: Training labels ``(N,)``.
        n: Subset size, a multiple of the number of classes.
        seed: Selects the subset.

    Returns:
        Sorted int64 indices ``(n,)``.
    """
    if n % NUM_CLASSES != 0:
        raise ValueError(f"n must be a multiple of {NUM_CLASSES}, got {n}")
    per_class = n // NUM_CLASSES
    generator = torch.Generator().manual_seed(seed)
    chosen = []
    for c in range(NUM_CLASSES):
        idx = (labels == c).nonzero().squeeze(1)
        if per_class > len(idx):
            raise ValueError(f"class {c} has only {len(idx)} examples, need {per_class}")
        chosen.append(idx[torch.randperm(len(idx), generator=generator)[:per_class]])
    return torch.cat(chosen).sort().values
