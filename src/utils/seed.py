"""Reproducibility helpers."""

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def step_generator(seed: int, step: int, stream: int = 0) -> torch.Generator:
    """A CPU generator determined by (seed, step, stream), so runs resume reproducibly."""
    return torch.Generator().manual_seed((seed * 1_000_003 + step) * 31 + stream)
