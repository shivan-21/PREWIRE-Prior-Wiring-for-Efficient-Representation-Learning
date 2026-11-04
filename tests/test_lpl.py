"""Parity of our LPL loss (src/learning_rules/lpl.py) with upstream's LPL module.

Runs upstream in the legacy env (.venv-upstream) on a fixed batch and compares losses and
gradients. Skipped where the legacy env is absent (e.g. on the cluster).
"""

import subprocess
import sys

import numpy as np
import pytest
import torch

from src.data.sequences import frame_pairs
from src.learning_rules.lpl import lpl_loss
from src.models.vgg import vgg11
from src.utils.paths import REPO_ROOT

LEGACY_PYTHON = (
    REPO_ROOT
    / ".venv-upstream"
    / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
)
pytestmark = pytest.mark.skipif(not LEGACY_PYTHON.exists(), reason="legacy env not installed")


@pytest.fixture(scope="module")
def reference(tmp_path_factory) -> dict[str, np.ndarray]:
    out = tmp_path_factory.mktemp("lpl_reference") / "reference.npz"
    script = REPO_ROOT / "tests" / "upstream" / "lpl_reference.py"
    subprocess.run(
        [str(LEGACY_PYTHON), "-I", str(script), str(REPO_ROOT), str(out)],
        check=True,
        capture_output=True,
    )
    with np.load(out) as data:
        return dict(data)


@pytest.mark.parametrize("tag", ["plain", "shuffled"])
def test_lpl_matches_upstream(reference, tag):
    encoder = vgg11(layer_local=True)
    state = {
        k.removeprefix("param/"): torch.from_numpy(v)
        for k, v in reference.items()
        if k.startswith("param/")
    }
    encoder.load_state_dict(state)
    x1, x2 = torch.from_numpy(reference["x1"]), torch.from_numpy(reference["x2"])
    if tag == "shuffled":
        # Our pipeline shuffles the second frames before the encoder; same result as upstream.
        x2 = x2[torch.from_numpy(reference[f"{tag}/perm"])]

    _, _, z1 = encoder(x1)
    _, _, z2 = encoder(x2)
    total, terms = lpl_loss(z1, z2)
    total.backward()

    # Gradients: summation order differs between torch versions, so compare at 1e-4 of each
    # gradient's largest element rather than elementwise (near-zero entries differ relatively).
    for name in ("pull", "push", "decorr"):
        np.testing.assert_allclose(
            terms[name].numpy(), reference[f"{tag}/{name}"], rtol=1e-4, atol=1e-7, err_msg=name
        )
    np.testing.assert_allclose(total.item(), reference[f"{tag}/total"], rtol=1e-5)
    np.testing.assert_allclose(
        encoder.blocks[0].module[0].weight.grad.numpy(),
        reference[f"{tag}/grad_first"],
        rtol=0,
        atol=1e-4 * abs(reference[f"{tag}/grad_first"]).max(),
    )
    np.testing.assert_allclose(
        encoder.blocks[7].module[0].weight.grad.numpy(),
        reference[f"{tag}/grad_last"],
        rtol=0,
        atol=1e-4 * abs(reference[f"{tag}/grad_last"]).max(),
    )


def test_frame_pairs_shuffle_equals_upstream_feature_shuffle():
    """Shuffling second frames before the encoder equals upstream's shuffle of z2 after it."""
    torch.manual_seed(0)
    encoder = vgg11(layer_local=True)
    frames = torch.randn(16, 2, 3, 32, 32)
    a, b = frame_pairs(frames, shuffle_positives=True, generator=torch.Generator().manual_seed(3))
    perm = torch.randperm(16, generator=torch.Generator().manual_seed(3))
    _, _, z_b = encoder(b)
    _, _, z_ref = encoder(frames[:, 1])
    for zb, zr in zip(z_b, z_ref, strict=True):
        torch.testing.assert_close(zb, zr[perm])
    torch.testing.assert_close(a, frames[:, 0])
