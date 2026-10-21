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


def data_root() -> Path:
    """Directory that holds one subdirectory per dataset (e.g. ``cifar10/``)."""
    return Path(os.environ.get(DATA_ROOT_ENV, DEFAULT_DATA_ROOT))
