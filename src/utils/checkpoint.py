"""Atomic checkpoint writes, so a job killed mid-save never leaves a corrupt file."""

import os
from pathlib import Path

import torch


def save_atomic(obj: object, path: str | Path) -> None:
    """``torch.save`` to a temporary file in the same directory, then rename over ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_if_exists(path: str | Path, map_location: str | torch.device = "cpu") -> object | None:
    """Load a checkpoint written by ``save_atomic``, or ``None`` if there is none."""
    path = Path(path)
    if not path.exists():
        return None
    return torch.load(path, map_location=map_location, weights_only=False)
