"""Tests for Oja/Sanger, SimCLR's loss, the schedule, configs and batch sampling."""

import math
import sys
from pathlib import Path

import pytest
import torch

from src.learning_rules.oja import OjaPretrainer, SangerLayer, extract_patches
from src.learning_rules.simclr import nt_xent_loss
from src.models.vgg import LAYER_DIMS, vgg11
from src.utils.config import apply_overrides, deep_merge, load_config
from src.utils.paths import REPO_ROOT
from src.utils.schedule import warmup_cosine_lr

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from pretrain import batch_indices  # noqa: E402


def test_sanger_recovers_principal_components():
    torch.manual_seed(0)
    dim, k = 27, 4  # one 3x3 RGB patch, 4 components
    basis = torch.linalg.qr(torch.randn(dim, dim))[0]
    scales = torch.cat((torch.tensor([4.0, 3.0, 2.0, 1.5]), torch.full((dim - k,), 0.5)))
    layer = SangerLayer(3, 2 * k, torch.device("cpu"), torch.Generator().manual_seed(1))
    for step in range(3000):
        x = (torch.randn(512, dim) * scales) @ basis.T + 5.0  # non-zero mean
        layer.update(x, lr=0.05 * (1 - step / 3000))
    cosines = (layer.weight @ basis[:, :k]).abs().diagonal()
    assert (cosines > 0.95).all(), cosines
    gram = layer.weight @ layer.weight.T
    torch.testing.assert_close(gram, torch.eye(k), atol=0.1, rtol=0)  # still converging


def test_sanger_conv_parameters_give_unit_variance_signed_pairs():
    torch.manual_seed(0)
    layer = SangerLayer(3, 8, torch.device("cpu"))
    x = torch.randn(4096, 27) * 2 + 1
    for _ in range(200):
        layer.update(x, lr=0.05)
    weight, bias = layer.conv_parameters()
    assert weight.shape == (8, 3, 3, 3) and bias.shape == (8,)
    y = x @ weight.view(8, -1).T + bias
    torch.testing.assert_close(y[:, :4], -y[:, 4:])
    assert torch.allclose(y.var(dim=0), torch.ones(8), atol=0.15)


def test_extract_patches_matches_conv():
    x = torch.randn(2, 3, 8, 8)
    weight = torch.randn(5, 3, 3, 3)
    patches = extract_patches(x)
    via_patches = (patches @ weight.view(5, -1).T).view(2, 64, 5).permute(0, 2, 1)
    torch.testing.assert_close(
        via_patches.reshape(2, 5, 8, 8), torch.nn.functional.conv2d(x, weight, padding=1)
    )


def test_oja_pretrainer_trains_every_layer():
    torch.manual_seed(0)
    encoder = vgg11(layer_local=True)
    before = [b.module[0].weight.clone() for b in encoder.blocks]
    trainer = OjaPretrainer(
        encoder, max_steps=16, lr=0.05, patches_per_step=2048, seed=0, device=torch.device("cpu")
    )
    for step in range(16):
        trainer.step(torch.randn(8, 3, 32, 32), step)
    for b, w0 in zip(encoder.blocks, before, strict=True):
        assert not torch.equal(b.module[0].weight, w0)
    assert [b.module[0].out_channels for b in encoder.blocks] == list(LAYER_DIMS)


def test_nt_xent_prefers_aligned_pairs():
    torch.manual_seed(0)
    h = torch.randn(32, 16)
    aligned = nt_xent_loss(h, h + 0.01 * torch.randn(32, 16))
    random = nt_xent_loss(h, torch.randn(32, 16))
    assert aligned < random
    # Lower bound: log(2N - 1) minus the best possible positive similarity / temperature.
    assert aligned > 0


def test_warmup_cosine_endpoints():
    assert warmup_cosine_lr(0, 1000, 1e-3, 10) == 0.0
    assert math.isclose(warmup_cosine_lr(10, 1000, 1e-3, 10), 1e-3)
    assert math.isclose(warmup_cosine_lr(1000, 1000, 1e-3, 10), 1e-6)
    lrs = [warmup_cosine_lr(s, 1000, 1e-3, 10) for s in range(10, 1001)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:], strict=False))


def test_config_merge_and_overrides(tmp_path: Path):
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}) == {"a": {"b": 3, "c": 2}}
    cfg = apply_overrides({"pretrain": {"lr": 1.0}}, ["pretrain.lr=1e-3", "pretrain.new=true"])
    assert cfg["pretrain"]["lr"] == 1e-3 and cfg["pretrain"]["new"] is True
    with pytest.raises(ValueError):
        apply_overrides({}, ["no_equals_sign"])


@pytest.mark.parametrize(
    "name", ["lpl", "simclr", "oja", "random_init", "lpl_shuffled", "lpl_temporal"]
)
def test_condition_configs_load(name):
    cfg = load_config(REPO_ROOT / "configs" / f"{name}.yaml")
    assert cfg["pretrain"]["method"] in ("lpl", "simclr", "oja", "random")
    assert cfg["pretrain"]["frames_per_step"] == 2048
    assert isinstance(cfg["pretrain"]["lr"], float)
    assert cfg["model"]["arch"] == "vgg11"


def test_lpl_temporal_inherits_budget():
    cfg = load_config(REPO_ROOT / "configs" / "lpl_temporal.yaml", ["pretrain.max_steps=123"])
    assert cfg["pretrain"]["max_steps"] == 123
    assert cfg["data"]["sequence"]["clip_length"] == 8


def test_batch_indices_cover_each_epoch_once_and_are_reproducible():
    per_step, n = 100, 1050
    epoch = torch.cat([batch_indices(s, per_step, n, seed=0) for s in range(10)])
    assert len(set(epoch.tolist())) == 1000  # 10 full batches, the 50 leftovers dropped
    torch.testing.assert_close(batch_indices(7, per_step, n, 0), batch_indices(7, per_step, n, 0))
    assert not torch.equal(batch_indices(0, per_step, n, 0), batch_indices(10, per_step, n, 0))
