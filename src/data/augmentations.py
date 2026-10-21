"""Upstream LPL's SimCLR augmentations, driven by explicit latent variables.

Upstream (``vendor/lpl_zenke/datasets/simclr_transforms.py``, ``SimCLRTrainDataTransform`` with
``input_height=32``) applies, in order:

1. RandomResizedCrop to 32x32: area scale U(0.08, 1), log aspect ratio U(log 3/4, log 4/3).
2. Horizontal flip, p = 0.5.
3. ColorJitter(0.8, 0.8, 0.8, 0.2) with p = 0.8: brightness, contrast and saturation factors
   U(0.2, 1.8), hue shift U(-0.2, 0.2), applied in a random order.
4. Grayscale, p = 0.2.
5. Gaussian blur, 3x3 kernel, sigma U(0.1, 2.0), p = 0.5.
6. CIFAR-10 normalisation.

Here every random choice is a deterministic function of standard-normal latents (a Gaussian
copula): ``q = Phi(u)`` is uniform on (0, 1) and goes through the parameter's inverse CDF; the
jitter order is the ranking of four latents. Independent latents reproduce upstream's
distribution. Correlated latents (see ``src/data/sequences.py``) give smooth sequences in which
every frame, taken alone, still has upstream's distribution.

Deliberate differences from upstream:

- Crops use continuous coordinates (no rounding to whole pixels), so crops can move smoothly.
  Which (scale, ratio) pairs are accepted still follows torchvision, including its rounding.
- Crops are resampled with bilinear ``roi_align``, which reads neighbouring pixels just outside the
  crop instead of clamping at the crop edge. This only affects the outermost output pixels.
- Everything runs on float tensors, so there is no uint8 rounding between steps.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

PARAM_NAMES = (
    "crop_log_ratio",
    "crop_scale",
    "crop_x",
    "crop_y",
    "flip",
    "jitter_on",
    "brightness",
    "contrast",
    "saturation",
    "hue",
    # Jitter order = ranking of these four latents. Independent normals give a uniformly random
    # permutation; correlated ones change the order only when two latents cross (rarely).
    "order_brightness",
    "order_contrast",
    "order_saturation",
    "order_hue",
    "grayscale",
    "blur_on",
    "blur_sigma",
)
NUM_PARAMS = len(PARAM_NAMES)
PARAM_INDEX = {name: i for i, name in enumerate(PARAM_NAMES)}

CROP_SCALE = (0.08, 1.0)
CROP_MAX_LOG_RATIO = math.log(4 / 3)  # symmetric range: log(3/4) = -log(4/3)
P_FLIP = 0.5
P_JITTER = 0.8
JITTER_FACTOR_RANGE = (0.2, 1.8)  # brightness, contrast, saturation
JITTER_HUE_RANGE = (-0.2, 0.2)
P_GRAYSCALE = 0.2
P_BLUR = 0.5
BLUR_SIGMA_RANGE = (0.1, 2.0)

# torchvision ColorJitter op ids: 0 brightness, 1 contrast, 2 saturation, 3 hue.
ORDER_PARAMS = ("order_brightness", "order_contrast", "order_saturation", "order_hue")

# pl_bolts cifar10_normalization
CIFAR10_MEAN = (125.3 / 255, 123.0 / 255, 113.9 / 255)
CIFAR10_STD = (63.0 / 255, 62.1 / 255, 66.7 / 255)

_ICDF_GRID_SIZE = 4097


@dataclass(frozen=True)
class AugmentationParams:
    """Per-image augmentation parameters, each with leading dimension N."""

    crop_boxes: torch.Tensor  # (N, 4) x0, y0, x1, y1 in pixel coordinates
    flip: torch.Tensor  # (N,) bool
    jitter_on: torch.Tensor  # (N,) bool
    jitter_factors: torch.Tensor  # (N, 4) brightness, contrast, saturation, hue
    jitter_order: torch.Tensor  # (N, 4) op ids in the order they are applied
    grayscale: torch.Tensor  # (N,) bool
    blur_on: torch.Tensor  # (N,) bool
    blur_sigma: torch.Tensor  # (N,)


def crop_max_scale(log_ratio: torch.Tensor, image_size: int = 32) -> torch.Tensor:
    """Largest crop scale torchvision accepts at a given log aspect ratio.

    torchvision draws scale s ~ U(0.08, 1) and log ratio l ~ U(-a, a), rounds the crop's width
    ``sqrt(s * r) * S`` and height ``sqrt(s / r) * S`` to whole pixels, and rejects crops whose
    rounded size exceeds the image. A side is accepted while it is at most ``S + 0.5`` pixels, so
    ``s <= c * exp(-|l|)`` with ``c = ((S + 0.5) / S) ** 2``, and never above 1.
    """
    c = ((image_size + 0.5) / image_size) ** 2
    return (c * torch.exp(-log_ratio.abs())).clamp(max=CROP_SCALE[1])


def _crop_log_ratio_table(image_size: int, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Grid of log ratios and the CDF of the accepted log ratio on it.

    Accepted log ratios have density proportional to ``crop_max_scale(l) - 0.08`` on [-a, a]
    (the length of the accepted scale interval). The CDF is its normalised trapezoid integral.
    """
    grid = torch.linspace(
        -CROP_MAX_LOG_RATIO, CROP_MAX_LOG_RATIO, _ICDF_GRID_SIZE, device=device, dtype=torch.float64
    )
    density = crop_max_scale(grid, image_size) - CROP_SCALE[0]
    steps = 0.5 * (density[1:] + density[:-1]) * (grid[1:] - grid[:-1])
    cdf = torch.cat((torch.zeros(1, device=device, dtype=grid.dtype), steps.cumsum(0)))
    return grid, cdf / cdf[-1]


def crop_log_ratio_cdf(log_ratio: torch.Tensor, image_size: int = 32) -> torch.Tensor:
    """CDF of the log aspect ratio of crops that torchvision accepts."""
    grid, cdf = _crop_log_ratio_table(image_size, log_ratio.device)
    x = log_ratio.to(grid.dtype).clamp(grid[0], grid[-1]).contiguous()
    hi = torch.searchsorted(grid, x).clamp(1, _ICDF_GRID_SIZE - 1)
    lo = hi - 1
    weight = (x - grid[lo]) / (grid[hi] - grid[lo])
    return (cdf[lo] + weight * (cdf[hi] - cdf[lo])).to(log_ratio.dtype)


def _crop_log_ratio_icdf(q: torch.Tensor, image_size: int) -> torch.Tensor:
    """Inverse of ``crop_log_ratio_cdf`` by linear interpolation on the same grid."""
    grid, cdf = _crop_log_ratio_table(image_size, q.device)
    q = q.to(cdf.dtype).contiguous()
    hi = torch.searchsorted(cdf, q).clamp(1, _ICDF_GRID_SIZE - 1)
    lo = hi - 1
    weight = (q - cdf[lo]) / (cdf[hi] - cdf[lo])
    return (grid[lo] + weight * (grid[hi] - grid[lo])).float()


def crop_scale_and_log_ratio(
    q_log_ratio: torch.Tensor, q_scale: torch.Tensor, image_size: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map two uniform variables to an accepted (scale, log ratio) pair.

    The log ratio follows its accepted marginal; given it, the scale is uniform on
    ``[0.08, crop_max_scale(log_ratio)]``. Together they are uniform over torchvision's accepted
    region, which is what its rejection sampling produces.
    """
    log_ratio = _crop_log_ratio_icdf(q_log_ratio, image_size)
    max_scale = crop_max_scale(log_ratio, image_size)
    return CROP_SCALE[0] + q_scale * (max_scale - CROP_SCALE[0]), log_ratio


def _uniform(q: torch.Tensor, bounds: tuple[float, float]) -> torch.Tensor:
    return bounds[0] + q * (bounds[1] - bounds[0])


def params_from_latents(latents: torch.Tensor, image_size: int = 32) -> AugmentationParams:
    """Map standard-normal latents to augmentation parameters.

    Args:
        latents: ``(N, NUM_PARAMS)``, one latent per parameter in ``PARAM_NAMES`` order. Each
            column should be marginally N(0, 1) for the parameters to follow upstream's
            distribution.
        image_size: Side length of the (square) input images.

    Returns:
        Parameters for ``apply_augmentations``.
    """
    if latents.shape[-1] != NUM_PARAMS:
        raise ValueError(f"expected {NUM_PARAMS} latents per image, got {latents.shape[-1]}")
    q = torch.special.ndtr(latents).clamp(1e-7, 1 - 1e-7)

    def col(name: str) -> torch.Tensor:
        return q[:, PARAM_INDEX[name]]

    scale, log_ratio = crop_scale_and_log_ratio(
        col("crop_log_ratio"), col("crop_scale"), image_size
    )
    ratio = torch.exp(log_ratio)
    # Sides up to S + 0.5 are accepted (torchvision rounds them to S); clamp them to the image.
    w = torch.sqrt(scale * ratio).clamp(max=1.0) * image_size
    h = torch.sqrt(scale / ratio).clamp(max=1.0) * image_size
    x0 = col("crop_x") * (image_size - w)
    y0 = col("crop_y") * (image_size - h)

    return AugmentationParams(
        crop_boxes=torch.stack((x0, y0, x0 + w, y0 + h), dim=1),
        flip=col("flip") < P_FLIP,
        jitter_on=col("jitter_on") < P_JITTER,
        jitter_factors=torch.stack(
            (
                _uniform(col("brightness"), JITTER_FACTOR_RANGE),
                _uniform(col("contrast"), JITTER_FACTOR_RANGE),
                _uniform(col("saturation"), JITTER_FACTOR_RANGE),
                _uniform(col("hue"), JITTER_HUE_RANGE),
            ),
            dim=1,
        ),
        jitter_order=latents[:, [PARAM_INDEX[name] for name in ORDER_PARAMS]].argsort(dim=1),
        grayscale=col("grayscale") < P_GRAYSCALE,
        blur_on=col("blur_on") < P_BLUR,
        blur_sigma=_uniform(col("blur_sigma"), BLUR_SIGMA_RANGE),
    )


# Colour ops follow torchvision.transforms.functional on float tensors in [0, 1].


def _grayscale(img: torch.Tensor) -> torch.Tensor:
    r, g, b = img.unbind(dim=1)
    return (0.2989 * r + 0.587 * g + 0.114 * b).unsqueeze(1)


def _brightness(img: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    return (factor * img).clamp(0, 1)


def _contrast(img: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    mean = _grayscale(img).mean(dim=(1, 2, 3), keepdim=True)
    return (factor * img + (1 - factor) * mean).clamp(0, 1)


def _saturation(img: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    return (factor * img + (1 - factor) * _grayscale(img)).clamp(0, 1)


def _rgb_to_hsv(img: torch.Tensor) -> torch.Tensor:
    r, g, b = img.unbind(dim=1)
    maxc = img.max(dim=1).values
    minc = img.min(dim=1).values
    eqc = maxc == minc
    cr = maxc - minc
    ones = torch.ones_like(maxc)
    s = cr / torch.where(eqc, ones, maxc)
    cr_divisor = torch.where(eqc, ones, cr)
    rc = (maxc - r) / cr_divisor
    gc = (maxc - g) / cr_divisor
    bc = (maxc - b) / cr_divisor
    hr = (maxc == r) * (bc - gc)
    hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
    hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
    h = torch.fmod((hr + hg + hb) / 6.0 + 1.0, 1.0)
    return torch.stack((h, s, maxc), dim=1)


def _hsv_to_rgb(img: torch.Tensor) -> torch.Tensor:
    h, s, v = img.unbind(dim=1)
    i = torch.floor(h * 6.0)
    f = h * 6.0 - i
    i = i.to(torch.int32) % 6
    p = (v * (1.0 - s)).clamp(0.0, 1.0)
    q = (v * (1.0 - s * f)).clamp(0.0, 1.0)
    t = (v * (1.0 - s * (1.0 - f))).clamp(0.0, 1.0)
    mask = (i.unsqueeze(1) == torch.arange(6, device=i.device).view(-1, 1, 1)).to(img.dtype)
    a1 = torch.stack((v, q, p, p, t, v), dim=1)
    a2 = torch.stack((t, v, v, q, p, p), dim=1)
    a3 = torch.stack((p, p, t, v, v, q), dim=1)
    a4 = torch.stack((a1, a2, a3), dim=1)  # (N, 3, 6, H, W)
    return torch.einsum("nijk,nxijk->nxjk", mask, a4)


def _hue(img: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    h, s, v = _rgb_to_hsv(img).unbind(dim=1)
    h = (h + shift.squeeze(1)) % 1.0
    return _hsv_to_rgb(torch.stack((h, s, v), dim=1))


_JITTER_OPS = (_brightness, _contrast, _saturation, _hue)


def _color_jitter(img: torch.Tensor, factors: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    """Apply the four jitter ops per image in that image's order (``(N, 4)`` op ids)."""
    factors = factors.view(-1, 4, 1, 1, 1)
    rows = torch.arange(img.shape[0], device=img.device)
    for step in range(4):
        candidates = torch.stack([op(img, factors[:, k]) for k, op in enumerate(_JITTER_OPS)])
        img = candidates[order[:, step], rows]
    return img


def _gaussian_blur3(img: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """3x3 Gaussian blur with per-image sigma and reflect padding (as cv2 and torchvision)."""
    edge = torch.exp(-0.5 / sigma**2).view(-1, 1, 1, 1)
    w_center = 1 / (1 + 2 * edge)
    w_edge = edge * w_center
    p = F.pad(img, (1, 1, 1, 1), mode="reflect")
    p = w_center * p[..., :, 1:-1] + w_edge * (p[..., :, :-2] + p[..., :, 2:])
    return w_center * p[..., 1:-1, :] + w_edge * (p[..., :-2, :] + p[..., 2:, :])


def _select(mask: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.where(mask.view(-1, 1, 1, 1), a, b)


def apply_augmentations(
    images: torch.Tensor, params: AugmentationParams, normalize: bool = True
) -> torch.Tensor:
    """Apply upstream's augmentation pipeline with the given parameters.

    Args:
        images: ``(N, 3, H, W)``, uint8 or float in [0, 1].
        params: One set of parameters per image.
        normalize: Apply CIFAR-10 normalisation at the end (as upstream does).

    Returns:
        Float tensor ``(N, 3, H, W)``.
    """
    if images.dtype == torch.uint8:
        images = images.float() / 255
    n, _, height, width = images.shape
    device = images.device

    boxes = torch.cat(
        (
            torch.arange(n, device=device, dtype=images.dtype).unsqueeze(1),
            params.crop_boxes.to(device, images.dtype),
        ),
        dim=1,
    )
    img = roi_align(images, boxes, output_size=(height, width), sampling_ratio=1, aligned=True)

    img = _select(params.flip.to(device), img.flip(-1), img)
    jittered = _color_jitter(
        img, params.jitter_factors.to(device, img.dtype), params.jitter_order.to(device)
    )
    img = _select(params.jitter_on.to(device), jittered, img)
    img = _select(params.grayscale.to(device), _grayscale(img).expand(-1, 3, -1, -1), img)
    img = _select(
        params.blur_on.to(device),
        _gaussian_blur3(img, params.blur_sigma.to(device, img.dtype)),
        img,
    )

    if normalize:
        mean = torch.tensor(CIFAR10_MEAN, device=device, dtype=img.dtype).view(1, 3, 1, 1)
        std = torch.tensor(CIFAR10_STD, device=device, dtype=img.dtype).view(1, 3, 1, 1)
        img = (img - mean) / std
    return img


def augment(images: torch.Tensor, latents: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Augment each image with the parameters given by its row of latents."""
    params = params_from_latents(latents.to(images.device), image_size=images.shape[-1])
    return apply_augmentations(images, params, normalize=normalize)
