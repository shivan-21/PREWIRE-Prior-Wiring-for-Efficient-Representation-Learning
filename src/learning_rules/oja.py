"""Oja-style Hebbian pre-training of VGG-11, one conv layer at a time.

Each conv layer learns with Sanger's generalized Hebbian algorithm (the multi-unit form of Oja's
rule), which converges to the top principal components of its input:

    y = W (x - m),    dW = eta * (y x^T - LT[y y^T] W)

``x`` is a 3x3 input patch (the conv's receptive field), ``m`` a running mean of patches and
``LT`` the lower triangle including the diagonal. Layers train greedily: layer k learns on the
outputs of layers 0..k-1, which are already trained and frozen.

To fit VGG-11's conv-ReLU-pool blocks:

- A layer with C output channels learns C/2 components and uses each with both signs, [W; -W].
  Principal components have arbitrary sign, so a ReLU after a single copy would discard half the
  signal at random. The first layer's 3x3 RGB patches have only 27 dimensions, so it learns 27
  components (54 channels) and leaves its other 10 channels at zero.
- Each component's output is rescaled to unit variance (homeostatic gain control). Without it,
  the activation scale grows with depth, since a patch carries ~9x the variance of one position.
- The step size is normalised by the mean squared patch norm, which bounds the largest eigenvalue,
  so one learning rate suits every layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN_MOMENTUM = 0.99
GAIN_EPS = 1e-6


def extract_patches(
    x: torch.Tensor, max_patches: int | None = None, generator: torch.Generator | None = None
) -> torch.Tensor:
    """3x3 patches with zero padding 1 (the conv's receptive fields), ``(num_patches, C * 9)``."""
    patches = F.unfold(x, kernel_size=3, padding=1)  # (N, C * 9, H * W)
    patches = patches.transpose(1, 2).reshape(-1, patches.shape[1])
    if max_patches is not None and patches.shape[0] > max_patches:
        idx = torch.randperm(patches.shape[0], generator=generator)[:max_patches]
        patches = patches[idx.to(patches.device)]
    return patches


class SangerLayer:
    """Sanger's rule for one conv layer, with running input mean and output variance."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> None:
        self.out_channels = out_channels
        self.dim = in_channels * 9
        # At most dim orthonormal components: the first layer (27-d RGB patches, 64 channels)
        # learns 27, and its remaining channels stay zero.
        self.num_components = min(out_channels // 2, self.dim)
        init = torch.randn(self.dim, self.num_components, generator=generator)
        self.weight = torch.linalg.qr(init)[0].T.contiguous().to(device)  # orthonormal rows
        self.mean = torch.zeros(self.dim, device=device)
        self.out_var = torch.ones(self.num_components, device=device)
        self.updates = 0

    def update(self, patches: torch.Tensor, lr: float) -> dict[str, float]:
        """One Sanger step on a batch of patches; returns monitoring metrics."""
        batch_mean = patches.mean(dim=0)
        if self.updates == 0:
            self.mean = batch_mean
        else:
            self.mean = MEAN_MOMENTUM * self.mean + (1 - MEAN_MOMENTUM) * batch_mean
        x = patches - self.mean
        y = x @ self.weight.T
        delta = (y.T @ x - torch.tril(y.T @ y) @ self.weight) / x.shape[0]
        eta = lr / x.pow(2).sum(dim=1).mean().clamp_min(GAIN_EPS)
        self.weight = self.weight + eta * delta
        var = y.pow(2).mean(dim=0)
        self.out_var = (
            var if self.updates == 0 else (MEAN_MOMENTUM * self.out_var + (1 - MEAN_MOMENTUM) * var)
        )
        self.updates += 1
        gram = self.weight @ self.weight.T
        eye = torch.eye(self.num_components, device=gram.device)
        return {
            "orthonormality_error": (gram - eye).norm().item() / self.num_components,
            # Variance captured along the (normalised) filter directions, as a fraction of total.
            "explained_variance": (
                (var / gram.diagonal()).sum() / x.pow(2).sum(dim=1).mean()
            ).item(),
        }

    def conv_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Conv weight ``(C, C_in, 3, 3)`` and bias ``(C,)`` computing [W; -W] (x - m) / gain.

        Channels beyond ``2 * num_components`` (first layer only) are zero.
        """
        w = self.weight / torch.sqrt(self.out_var + GAIN_EPS).unsqueeze(1)
        unused = self.out_channels - 2 * self.num_components
        w = torch.cat((w, -w, w.new_zeros(unused, self.dim)))
        return w.view(w.shape[0], -1, 3, 3), -(w @ self.mean)

    def state_dict(self) -> dict:
        return {
            "weight": self.weight,
            "mean": self.mean,
            "out_var": self.out_var,
            "updates": self.updates,
        }

    def load_state_dict(self, state: dict) -> None:
        self.weight, self.mean = state["weight"], state["mean"]
        self.out_var, self.updates = state["out_var"], state["updates"]


class OjaPretrainer:
    """Greedy layer-by-layer Sanger training of a layer-local VGG-11 encoder.

    The step budget is split equally across the 8 layers. ``step(frames, global_step)`` trains the
    current layer and writes its conv parameters into the encoder when its share of steps is done.
    """

    def __init__(
        self,
        encoder: nn.Module,
        max_steps: int,
        lr: float,
        patches_per_step: int,
        seed: int,
        device: torch.device,
    ) -> None:
        self.encoder = encoder
        self.num_layers = len(encoder.blocks)
        self.max_steps = max_steps
        self.steps_per_layer = max(1, max_steps // self.num_layers)
        self.lr = lr
        self.patches_per_step = patches_per_step
        self.seed = seed
        self.device = device
        self.layer: SangerLayer | None = None
        self.layer_index = -1

    def _conv(self, k: int) -> nn.Conv2d:
        return self.encoder.blocks[k].module[0]

    def _start_layer(self, k: int) -> None:
        conv = self._conv(k)
        generator = torch.Generator().manual_seed(self.seed * 100 + k)
        self.layer = SangerLayer(conv.in_channels, conv.out_channels, self.device, generator)
        self.layer_index = k

    def step(self, frames: torch.Tensor, global_step: int) -> dict[str, float]:
        k = min(global_step // self.steps_per_layer, self.num_layers - 1)
        if k != self.layer_index:
            self._start_layer(k)
        step_in_layer = global_step - k * self.steps_per_layer
        # The last layer also takes the remainder of max_steps / num_layers.
        layer_steps = (
            self.steps_per_layer
            if k < self.num_layers - 1
            else max(1, self.max_steps - k * self.steps_per_layer)
        )
        # Within each layer: linear decay of the step size to zero (stochastic approximation).
        lr = self.lr * (1 - step_in_layer / layer_steps)

        with torch.no_grad():
            x = frames
            for block in self.encoder.blocks[:k]:
                x = block(x)
            generator = torch.Generator().manual_seed(self.seed * 1_000_003 + global_step)
            patches = extract_patches(x, self.patches_per_step, generator)
            metrics = self.layer.update(patches, lr)
            last_step_of_layer = step_in_layer == layer_steps - 1
            if last_step_of_layer or global_step % 100 == 0:
                weight, bias = self.layer.conv_parameters()
                self._conv(k).weight.copy_(weight)
                self._conv(k).bias.copy_(bias)
        return {"layer": float(k), "lr": lr, **metrics}

    def state_dict(self) -> dict:
        return {
            "layer_index": self.layer_index,
            "layer": None if self.layer is None else self.layer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state["layer"] is not None:
            self._start_layer(state["layer_index"])
            self.layer.load_state_dict(state["layer"])
