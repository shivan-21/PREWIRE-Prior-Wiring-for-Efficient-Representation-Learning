"""Tests for the latent-driven augmentations and the synthetic sequences."""

import itertools
import math

import pytest
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from scipy import stats

from src.data import augmentations as aug
from src.data.sequences import ar1_latents, frame_pairs, make_clips, shuffle_frames

NUM_CLIPS = 20_000
RHOS = (0.0, 0.8, 0.95)
# About 40 seeded statistical checks, so a strict per-check threshold keeps the chance of a false
# failure low; with 20,000 samples a real mismatch gives p-values far below it.
P_MIN = 1e-4


def final_frame_params(rho: float) -> aug.AugmentationParams:
    """Parameters of the last frame of many independent clips (independent across clips)."""
    generator = torch.Generator().manual_seed(0)
    latents = ar1_latents(NUM_CLIPS, 8, rho, generator=generator)[:, -1]
    return aug.params_from_latents(latents)


@pytest.mark.parametrize("rho", RHOS)
def test_continuous_marginals_match_upstream(rho):
    p = final_frame_params(rho)
    checks = {
        "brightness": (p.jitter_factors[:, 0], aug.JITTER_FACTOR_RANGE),
        "contrast": (p.jitter_factors[:, 1], aug.JITTER_FACTOR_RANGE),
        "saturation": (p.jitter_factors[:, 2], aug.JITTER_FACTOR_RANGE),
        "hue": (p.jitter_factors[:, 3], aug.JITTER_HUE_RANGE),
        "blur_sigma": (p.blur_sigma, aug.BLUR_SIGMA_RANGE),
    }
    for name, (values, (lo, hi)) in checks.items():
        assert stats.kstest(values.numpy(), "uniform", args=(lo, hi - lo)).pvalue > P_MIN, name


@pytest.mark.parametrize("rho", RHOS)
def test_switch_probabilities_match_upstream(rho):
    p = final_frame_params(rho)
    checks = {
        "flip": (p.flip, aug.P_FLIP),
        "jitter_on": (p.jitter_on, aug.P_JITTER),
        "grayscale": (p.grayscale, aug.P_GRAYSCALE),
        "blur_on": (p.blur_on, aug.P_BLUR),
    }
    for name, (mask, prob) in checks.items():
        assert stats.binomtest(int(mask.sum()), NUM_CLIPS, prob).pvalue > P_MIN, name
    # Jitter order: uniform over all 24 permutations.
    _, counts = torch.unique(p.jitter_order, dim=0, return_counts=True)
    assert len(counts) == 24
    assert stats.chisquare(counts.numpy()).pvalue > P_MIN


def order_change_rate(rho: float) -> float:
    """Fraction of consecutive frames whose jitter order differs."""
    latents = ar1_latents(2_000, 8, rho, generator=torch.Generator().manual_seed(6))
    order = aug.params_from_latents(latents.reshape(-1, aug.NUM_PARAMS)).jitter_order
    order = order.view(2_000, 8, 4)
    return (order[:, 1:] != order[:, :-1]).any(dim=-1).float().mean().item()


def test_jitter_order_changes_less_when_smooth():
    # Rates: about 0.96 at rho = 0 (independent frames), 0.46 at rho = 0.95.
    rates = [order_change_rate(rho) for rho in (0.0, 0.8, 0.95)]
    assert abs(rates[0] - 23 / 24) < 0.02
    assert rates[0] > rates[1] > rates[2]
    assert rates[2] < 0.5 * rates[0]


@pytest.mark.parametrize("rho", RHOS)
def test_crop_distribution_matches_accepted_region(rho):
    boxes = final_frame_params(rho).crop_boxes
    x0, y0, x1, y1 = boxes.unbind(dim=1)
    w = x1 - x0
    assert (x0 >= 0).all() and (y0 >= 0).all()
    assert (x1 <= 32 + 1e-4).all() and (y1 <= 32 + 1e-4).all()

    generator = torch.Generator().manual_seed(0)
    latents = ar1_latents(NUM_CLIPS, 8, rho, generator=generator)[:, -1]
    q = torch.special.ndtr(latents)
    scale, log_ratio = aug.crop_scale_and_log_ratio(
        q[:, aug.PARAM_INDEX["crop_log_ratio"]], q[:, aug.PARAM_INDEX["crop_scale"]]
    )
    ks = stats.kstest(
        log_ratio.numpy(), lambda v: aug.crop_log_ratio_cdf(torch.as_tensor(v)).numpy()
    )
    assert ks.pvalue > P_MIN

    # Given the ratio, the scale is uniform on [0.08, crop_max_scale(log ratio)].
    relative = (scale - aug.CROP_SCALE[0]) / (aug.crop_max_scale(log_ratio) - aug.CROP_SCALE[0])
    assert stats.kstest(relative.numpy(), "uniform").pvalue > P_MIN

    # Position is uniform over the free space.
    free = 32 - w
    usable = free > 1e-3
    assert stats.kstest((x0[usable] / free[usable]).numpy(), "uniform").pvalue > P_MIN


def test_crop_acceptance_matches_torchvision_rejection_rule():
    """Simulate torchvision's rejection sampling (continuous values, rounded acceptance)."""
    g = torch.Generator().manual_seed(7)
    n = 60_000
    s = aug.CROP_SCALE[0] + torch.rand(n, generator=g, dtype=torch.float64) * 0.92
    lr = (torch.rand(n, generator=g, dtype=torch.float64) * 2 - 1) * aug.CROP_MAX_LOG_RATIO
    w = torch.round(torch.sqrt(s * torch.exp(lr)) * 32)
    h = torch.round(torch.sqrt(s / torch.exp(lr)) * 32)
    keep = (w <= 32) & (h <= 32)
    q = torch.rand(2, NUM_CLIPS, generator=g)
    ours_s, ours_lr = aug.crop_scale_and_log_ratio(q[0], q[1])
    assert stats.ks_2samp(lr[keep].numpy(), ours_lr.numpy()).pvalue > P_MIN
    assert stats.ks_2samp(s[keep].numpy(), ours_s.numpy()).pvalue > P_MIN


def test_crop_scale_close_to_torchvision():
    torch.manual_seed(0)
    image = torch.zeros(3, 32, 32)
    tv = torch.tensor(
        [
            math.prod(T.RandomResizedCrop.get_params(image, [0.08, 1.0], [3 / 4, 4 / 3])[2:]) / 1024
            for _ in range(20_000)
        ]
    )
    ours = final_frame_params(0.0).crop_boxes
    ours = (ours[:, 2] - ours[:, 0]) * (ours[:, 3] - ours[:, 1]) / 1024
    # Same accepted region as torchvision; only its rounding of crops to whole pixels differs.
    assert abs(tv.mean() - ours.mean()) < 0.005
    assert abs(tv.std() - ours.std()) < 0.005


@pytest.mark.parametrize("rho", RHOS)
def test_ar1_lag_correlation_and_stationarity(rho):
    generator = torch.Generator().manual_seed(1)
    latents = ar1_latents(NUM_CLIPS, 8, rho, num_params=1, generator=generator)[..., 0]
    assert abs(latents[:, -1].var().item() - 1.0) < 0.05
    for gap in (1, 2, 4):
        corr = torch.corrcoef(torch.stack((latents[:, 0], latents[:, gap])))[0, 1].item()
        assert abs(corr - rho**gap) < 0.03, gap


def test_shuffle_frames_keeps_each_clips_frames():
    latents = ar1_latents(100, 8, 0.9, generator=torch.Generator().manual_seed(2))
    shuffled = shuffle_frames(latents, generator=torch.Generator().manual_seed(3))
    key = 0  # sort each clip's frames by one latent and compare
    order_a = latents[..., key].argsort(dim=1).unsqueeze(-1)
    order_b = shuffled[..., key].argsort(dim=1).unsqueeze(-1)
    torch.testing.assert_close(
        torch.take_along_dim(latents, order_a, dim=1),
        torch.take_along_dim(shuffled, order_b, dim=1),
    )
    assert not torch.equal(latents, shuffled)


def test_frame_pairs():
    x = torch.arange(3 * 5).view(3, 5)  # clip c, frame t -> 5c + t
    z1, z2 = frame_pairs(x, frame_gap=2)
    assert z1.shape == z2.shape == (9,)
    torch.testing.assert_close(z2 - z1, torch.full((9,), 2))
    assert ((z1 // 5) == (z2 // 5)).all()  # same clip

    _, z2_shuffled = frame_pairs(
        x, shuffle_positives=True, generator=torch.Generator().manual_seed(0)
    )
    assert sorted(z2_shuffled.tolist()) == sorted(frame_pairs(x)[1].tolist())

    with pytest.raises(ValueError):
        frame_pairs(x, frame_gap=5)


@pytest.fixture
def images() -> torch.Tensor:
    return torch.rand(6, 3, 32, 32, generator=torch.Generator().manual_seed(4))


def test_color_ops_match_torchvision(images):
    factors = (0.2, 0.5, 1.0, 1.3, 1.8, 0.9)
    shifts = (-0.2, -0.1, 0.0, 0.05, 0.15, 0.2)
    for img, f, s in zip(images, factors, shifts, strict=True):
        batch, f_t, s_t = (
            img.unsqueeze(0),
            torch.tensor(f).view(1, 1, 1, 1),
            torch.tensor(s).view(1, 1, 1, 1),
        )
        torch.testing.assert_close(aug._brightness(batch, f_t)[0], TF.adjust_brightness(img, f))
        torch.testing.assert_close(aug._contrast(batch, f_t)[0], TF.adjust_contrast(img, f))
        torch.testing.assert_close(aug._saturation(batch, f_t)[0], TF.adjust_saturation(img, f))
        torch.testing.assert_close(
            aug._hue(batch, s_t)[0], TF.adjust_hue(img, s), atol=1e-5, rtol=0
        )
    torch.testing.assert_close(
        aug._grayscale(images).expand(-1, 3, -1, -1),
        TF.rgb_to_grayscale(images, num_output_channels=3),
    )


def test_blur_matches_torchvision(images):
    sigmas = torch.tensor([0.1, 0.5, 1.0, 1.5, 2.0, 0.7])
    ours = aug._gaussian_blur3(images, sigmas)
    for img, out, sigma in zip(images, ours, sigmas, strict=True):
        torch.testing.assert_close(out, TF.gaussian_blur(img, [3, 3], [sigma.item()] * 2))


def test_jitter_order_is_applied(images):
    factors = torch.tensor([[1.5, 0.5, 1.7, 0.1]]).expand(len(images), 4)
    orders = torch.tensor(list(itertools.permutations(range(4))))[:: 24 // len(images)]
    ours = aug._color_jitter(images, factors, orders)
    fns = (TF.adjust_brightness, TF.adjust_contrast, TF.adjust_saturation, TF.adjust_hue)
    for img, out, order in zip(images, ours, orders, strict=True):
        expected = img
        for op in order.tolist():
            expected = fns[op](expected, factors[0, op].item())
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=0)


def test_crop_matches_torchvision_resized_crop(images):
    top, left, height, width = 4, 6, 16, 20
    params = aug.params_from_latents(torch.zeros(len(images), aug.NUM_PARAMS))
    params = aug.AugmentationParams(
        crop_boxes=torch.tensor(
            [[left, top, left + width, top + height]], dtype=torch.float
        ).expand(len(images), 4),
        flip=torch.zeros(len(images), dtype=torch.bool),
        jitter_on=torch.zeros(len(images), dtype=torch.bool),
        jitter_factors=params.jitter_factors,
        jitter_order=params.jitter_order,
        grayscale=torch.zeros(len(images), dtype=torch.bool),
        blur_on=torch.zeros(len(images), dtype=torch.bool),
        blur_sigma=params.blur_sigma,
    )
    ours = aug.apply_augmentations(images, params, normalize=False)
    expected = TF.resized_crop(images, top, left, height, width, [32, 32], antialias=False)
    # Outermost pixels differ by design: roi_align reads pixels just outside the crop.
    torch.testing.assert_close(ours[..., 1:-1, 1:-1], expected[..., 1:-1, 1:-1])


def test_all_off_latents_give_identity(images):
    latents = torch.full((len(images), aug.NUM_PARAMS), 10.0)  # q -> 1: every switch off
    latents[:, aug.PARAM_INDEX["crop_log_ratio"]] = 0.0  # square crop; scale q -> 1: whole image
    out = aug.augment(images, latents, normalize=False)
    torch.testing.assert_close(out, images, atol=1e-4, rtol=0)


def test_clips_shape_and_smoothness(images):
    uint8_images = (images * 255).to(torch.uint8)
    errors = {}
    for rho in (0.0, 0.95):
        clips = make_clips(
            uint8_images, clip_length=8, rho=rho, generator=torch.Generator().manual_seed(5)
        )
        assert clips.shape == (6, 8, 3, 32, 32)
        assert torch.isfinite(clips).all()
        errors[rho] = (clips[:, 1:] - clips[:, :-1]).pow(2).mean().item()
    assert errors[0.95] < 0.5 * errors[0.0]
