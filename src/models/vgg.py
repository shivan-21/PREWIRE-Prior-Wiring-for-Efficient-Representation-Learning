"""Shared VGG-11 backbone: upstream LPL's ``VGG11Encoder``.

Upstream: ``vendor/lpl_zenke/models/encoders.py``.

The module is loaded by file path rather than by putting ``vendor/lpl_zenke`` on ``sys.path``,
which would expose upstream's generic top-level names (``models``, ``utils``, ``datasets``).

The encoder has 8 conv blocks (conv 3x3, ReLU, optional max-pool), no batch norm. Calling it
returns ``(pooled, feature_maps, z)``: ``pooled`` is the global-average-pooled last block
``(N, 512)``; ``z`` is a list of pooled block outputs (all 8 blocks when layer-local, only the last
otherwise). Layer-local mode detaches each block's input, so no gradient crosses blocks. Both modes
have identical parameters, so weights trained one way load into the other.
"""

import importlib.util
from functools import cache
from pathlib import Path
from types import ModuleType

import torch
import torch.nn as nn

from src.utils.paths import REPO_ROOT

UPSTREAM_ENCODERS = REPO_ROOT / "vendor" / "lpl_zenke" / "models" / "encoders.py"
LAYER_DIMS = (64, 128, 256, 256, 512, 512, 512, 512)
FEATURE_DIM = LAYER_DIMS[-1]


@cache
def _upstream_encoders() -> ModuleType:
    spec = importlib.util.spec_from_file_location("lpl_zenke_encoders", UPSTREAM_ENCODERS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def kaiming_init_(encoder: nn.Module) -> nn.Module:
    """He-normal conv weights (fan-in, ReLU gain) and zero biases, in place.

    PyTorch's default conv init (which upstream uses) shrinks activation variance about 6x per
    conv-ReLU layer, so after 8 layers the output is ~1e-3 of the input. End-to-end SimCLR then
    collapses (all projections equal), and the random-init baseline is degenerate. He init keeps
    the variance roughly constant with depth. Every condition starts from this init.
    """
    for module in encoder.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    return encoder


def vgg11(layer_local: bool, init: bool = True) -> nn.Module:
    """Upstream VGG-11 encoder for 32x32 inputs, without projector MLPs.

    Args:
        layer_local: Detach between blocks and return all 8 pooled block outputs (LPL, Oja).
            ``False`` trains end to end and returns only the last (SimCLR, fine-tuning).
        init: Apply ``kaiming_init_``. ``False`` keeps PyTorch's default init (as upstream),
            e.g. before loading saved weights.
    """
    encoder = _upstream_encoders().VGG11Encoder(train_end_to_end=not layer_local)
    return kaiming_init_(encoder) if init else encoder


def load_encoder(path: str | Path, layer_local: bool, device: torch.device) -> nn.Module:
    """VGG-11 with weights from a pre-training output (``encoder.pt`` or a snapshot).

    Weights are mode-independent, so any run can be loaded layer-local (per-layer probes) or end
    to end (fine-tuning).
    """
    state = torch.load(path, map_location=device, weights_only=False)["encoder"]
    encoder = vgg11(layer_local=layer_local, init=False).to(device)
    encoder.load_state_dict(state)
    return encoder
