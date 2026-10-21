"""Render example synthetic clips for checking by eye.

One block of rows per image, one row per rho, one column per frame. Saves a PDF (and a PNG for
quick inspection) to results_and_figs/figures/.

    python scripts/preview_clips.py --rhos 0 0.5 0.8 0.95
"""

import argparse
from datetime import date

import matplotlib.pyplot as plt
import torch

from src.data.cifar import load_cifar10
from src.data.sequences import make_clips
from src.utils.paths import REPO_ROOT

CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--num_images", type=int, default=3)
    parser.add_argument("--clip_length", type=int, default=8)
    parser.add_argument("--rhos", type=float, nargs="+", default=[0.0, 0.5, 0.8, 0.95])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default=str(REPO_ROOT / "results_and_figs" / "figures"))
    args = parser.parse_args()

    images, labels = load_cifar10("train")
    # First image of each of the first num_images classes, for variety.
    indices = [int((labels == c).nonzero()[0]) for c in range(args.num_images)]

    rows = []  # (image index, rho, frames (L, 3, H, W))
    for i in indices:
        for rho in args.rhos:
            generator = torch.Generator().manual_seed(args.seed)
            clip = make_clips(
                images[i : i + 1], args.clip_length, rho, generator=generator, normalize=False
            )[0]
            rows.append((i, rho, clip))

    fig, axes = plt.subplots(
        len(rows),
        args.clip_length + 1,
        figsize=(1.1 * (args.clip_length + 1), 1.1 * len(rows)),
        squeeze=False,
    )
    for row, (i, rho, clip) in zip(axes, rows, strict=True):
        row[0].imshow(images[i].permute(1, 2, 0))
        row[0].set_ylabel(f"rho={rho:g}", rotation=0, ha="right", va="center", fontsize=8)
        for ax, frame in zip(row[1:], clip, strict=True):
            ax.imshow(frame.permute(1, 2, 0).clamp(0, 1))
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, 0].set_title("original", fontsize=8)
    for t in range(args.clip_length):
        axes[0, t + 1].set_title(f"t={t}", fontsize=8)
    fig.suptitle(
        f"Synthetic clips ({', '.join(CLASS_NAMES[c] for c in range(args.num_images))})", fontsize=9
    )
    fig.tight_layout()

    stem = f"{args.out_dir}/temporal_clips_preview_{date.today():%Y%m%d}"
    fig.savefig(f"{stem}.pdf", format="pdf")
    fig.savefig(f"{stem}.png", dpi=120)
    print(f"saved {stem}.pdf and .png")


if __name__ == "__main__":
    main()
