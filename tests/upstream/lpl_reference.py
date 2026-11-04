"""Upstream LPL losses and gradients on a fixed batch, for the parity test (run in .venv-upstream).

    python -I tests/upstream/lpl_reference.py <repo_root> <out.npz>

Saves the encoder weights, the batch, and for plain and shuffled positives: the per-layer pull,
push and decorrelation losses from ``LPL.loss_step``, the permutation it used, and the gradients
of the first and last conv weights for the total loss of ``LPL.training_step``.
"""

import sys
from pathlib import Path

import numpy as np
import torch

repo_root, out_path = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(repo_root / "vendor" / "lpl_zenke"))
from models.module import LPL  # noqa: E402

BATCH = 64

torch.manual_seed(0)
model = LPL(num_classes=10)
generator = torch.Generator().manual_seed(1)
x1 = torch.randn(BATCH, 3, 32, 32, generator=generator)
x2 = x1 + 0.5 * torch.randn(BATCH, 3, 32, 32, generator=generator)

results = {f"param/{k}": v.numpy() for k, v in model.network.encoder.state_dict().items()}
results["x1"], results["x2"] = x1.numpy(), x2.numpy()
blocks = model.network.encoder.blocks

for tag, shuffle in (("plain", False), ("shuffled", True)):
    model.hparams.shuffle_positives = shuffle
    model.zero_grad()
    torch.manual_seed(2)
    perm = torch.randperm(BATCH)
    torch.manual_seed(
        2
    )  # loss_step's randperm is the first draw after this (forward is deterministic)
    pull, push, decorr = model.loss_step(((x1, x2, None), None))
    total = (
        model.hparams.pull_coeff * pull.sum()
        + model.hparams.push_coeff * push.sum()
        + model.hparams.decorr_coeff * decorr.sum()
    )
    total.backward()
    results[f"{tag}/pull"] = pull.detach().numpy()
    results[f"{tag}/push"] = push.detach().numpy()
    results[f"{tag}/decorr"] = decorr.detach().numpy()
    results[f"{tag}/total"] = total.detach().numpy()
    results[f"{tag}/perm"] = perm.numpy()
    results[f"{tag}/grad_first"] = blocks[0].module[0].weight.grad.numpy().copy()
    results[f"{tag}/grad_last"] = blocks[7].module[0].weight.grad.numpy().copy()

np.savez(out_path, **results)
