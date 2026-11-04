"""Data and output roots.

Each root comes from an environment variable when set (the cluster sets them; see CLUSTER.md),
otherwise from a repo-relative default.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT_ENV = "PREWIRE_DATA"
# Shared with upstream LPL, which reads ~/data/datasets with HOME=data/upstream_home (see README).
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "upstream_home" / "data" / "datasets"


OUT_ROOT_ENV = "PREWIRE_OUT"
DEFAULT_OUT_ROOT = REPO_ROOT / "results_and_figs" / "checkpoints"


def data_root() -> Path:
    """Directory that holds one subdirectory per dataset (e.g. ``cifar10/``)."""
    return Path(os.environ.get(DATA_ROOT_ENV, DEFAULT_DATA_ROOT))


def out_root() -> Path:
    """Directory for run outputs: one subdirectory per run (checkpoints, metrics)."""
    return Path(os.environ.get(OUT_ROOT_ENV, DEFAULT_OUT_ROOT))
