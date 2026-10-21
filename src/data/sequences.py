"""Synthetic CIFAR-10 sequences: smooth trajectories through upstream's augmentations.

Each clip follows one image for ``clip_length`` frames. Every augmentation latent (see
``src/data/augmentations.py``) follows a stationary AR(1) process::

    u[t + 1] = rho * u[t] + sqrt(1 - rho**2) * eps,    u[0], eps ~ N(0, 1)

so every frame is marginally N(0, 1), i.e. has exactly upstream's augmentation distribution, and
frames ``dt`` apart have latent correlation ``rho**dt`` (correlation time ``-1 / ln(rho)`` frames).
``rho = 0`` gives independent upstream views; a 2-frame clip with ``rho = 0`` is upstream LPL's
two-view pair.
"""

import math

import torch

from src.data.augmentations import NUM_PARAMS, augment

ORDERS = ("temporal", "shuffled")


def ar1_latents(
    num_clips: int,
    clip_length: int,
    rho: float,
    num_params: int = NUM_PARAMS,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample stationary AR(1) latent trajectories.

    Args:
        num_clips: Number of independent clips.
        clip_length: Frames per clip.
        rho: Frame-to-frame correlation, in [0, 1].
        num_params: Latents per frame.
        generator: Optional CPU random generator.

    Returns:
        ``(num_clips, clip_length, num_params)`` on the CPU.
    """
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [0, 1], got {rho}")
    noise = torch.randn(num_clips, clip_length, num_params, generator=generator)
    latents = torch.empty_like(noise)
    latents[:, 0] = noise[:, 0]
    innovation_scale = math.sqrt(1.0 - rho**2)
    for t in range(1, clip_length):
        latents[:, t] = rho * latents[:, t - 1] + innovation_scale * noise[:, t]
    return latents


def shuffle_frames(latents: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Randomly reorder the frames of each clip (independently per clip)."""
    num_clips, clip_length = latents.shape[:2]
    perm = torch.rand(num_clips, clip_length, generator=generator).argsort(dim=1)
    return torch.take_along_dim(latents, perm.unsqueeze(-1), dim=1)


def make_clips(
    images: torch.Tensor,
    clip_length: int,
    rho: float,
    order: str = "temporal",
    generator: torch.Generator | None = None,
    normalize: bool = True,
) -> torch.Tensor:
    """Turn each image into a clip of augmented frames.

    Args:
        images: ``(C, 3, H, W)``, uint8 or float in [0, 1]; one clip per image.
        clip_length: Frames per clip.
        rho: Frame-to-frame latent correlation (0 = independent upstream views).
        order: ``"temporal"`` keeps the AR(1) order; ``"shuffled"`` reorders each clip's frames
            randomly, keeping the same set of frames.
        generator: Optional CPU random generator for the latents.
        normalize: Apply CIFAR-10 normalisation.

    Returns:
        ``(C, clip_length, 3, H, W)`` float frames on the images' device.
    """
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}, got {order!r}")
    num_clips = images.shape[0]
    latents = ar1_latents(num_clips, clip_length, rho, generator=generator)
    if order == "shuffled":
        latents = shuffle_frames(latents, generator=generator)
    frames = augment(
        images.repeat_interleave(clip_length, dim=0),
        latents.reshape(num_clips * clip_length, NUM_PARAMS),
        normalize=normalize,
    )
    return frames.reshape(num_clips, clip_length, *frames.shape[1:])


def frame_pairs(
    x: torch.Tensor,
    frame_gap: int = 1,
    shuffle_positives: bool = False,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pair frames ``frame_gap`` apart within each clip, for LPL's pull loss.

    Works on frames or on per-frame features: ``x`` is ``(C, L, ...)``. Mirrors upstream's
    ``LPL.loss_step``, where ``z1`` and ``z2`` are the two members of each pair.

    Args:
        x: Clips of frames or features, ``(C, L, ...)``.
        frame_gap: Distance in frames between the members of a pair.
        shuffle_positives: Pair each ``z1`` with a random other pair's ``z2`` (upstream's
            ``--shuffle_positives``), removing the same-image pairing.
        generator: Optional CPU random generator for the shuffle.

    Returns:
        ``(z1, z2)``, each ``(C * (L - frame_gap), ...)``.
    """
    if not 1 <= frame_gap < x.shape[1]:
        raise ValueError(f"frame_gap must be in [1, {x.shape[1] - 1}], got {frame_gap}")
    rest = x.shape[2:]
    z1 = x[:, :-frame_gap].reshape(-1, *rest)
    z2 = x[:, frame_gap:].reshape(-1, *rest)
    if shuffle_positives:
        z2 = z2[torch.randperm(z2.shape[0], generator=generator).to(z2.device)]
    return z1, z2
