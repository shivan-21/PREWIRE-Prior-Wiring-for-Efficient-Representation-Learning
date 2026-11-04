"""Phase 2: supervised fine-tuning of a pre-trained VGG-11 on a labelled subset, then test accuracy.

    python scripts/finetune.py --encoders lpl=results_and_figs/checkpoints/lpl_s0 --seed 0

For every encoder and labelled-set size in ``finetune.labeled_counts``, the whole network plus a
linear head is trained on the class-balanced labelled subset (``src/data/cifar.labeled_subset``)
and scored on the 10k test images. One fixed recipe for every condition:

- Pooled 512-d features pass through a fixed standardisation (mean/std over all 50k unlabelled
  training images under the starting weights) before the head, so no method starts with
  exploding or vanishing logits. It is frozen; it changes conditioning, not capacity.
- AdamW (lr 3e-4, weight decay 0.05), cosine decay to 0, batch min(256, n), standard CIFAR
  augmentation (4-pixel padded random crop, horizontal flip).
- Steps: max(1000, 30 epochs), so tiny sets still get enough updates. No early stopping and no
  validation split: the final model is scored.

Results are appended to ``results_and_figs/tables/finetune_s<seed>.csv`` and logged to MLflow
(``finetune/<name>_n<n>``).
"""

import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.analysis.linear_probe import extract_features, normalize_images
from src.data.cifar import NUM_CLASSES, labeled_subset, load_cifar10
from src.models.vgg import FEATURE_DIM, load_encoder
from src.utils.config import load_config
from src.utils.logging import RunLogger
from src.utils.paths import REPO_ROOT, out_root
from src.utils.seed import seed_everything

LR = 3e-4
WEIGHT_DECAY = 0.05
MAX_BATCH = 256
MIN_STEPS = 1000
EPOCHS = 30


class Classifier(nn.Module):
    """Encoder, frozen feature standardisation, linear head."""

    def __init__(self, encoder: nn.Module, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.encoder = encoder
        self.register_buffer("mean", mean)
        self.register_buffer("std", std.clamp_min(1e-6))
        self.head = nn.Linear(FEATURE_DIM, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.encoder(x)[0]
        return self.head((pooled - self.mean) / self.std)


def augment(images: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """4-pixel padded random crop and horizontal flip on uint8 ``(B, 3, 32, 32)`` (on device)."""
    b = images.shape[0]
    padded = F.pad(images, (4, 4, 4, 4))
    dx = torch.randint(0, 9, (b,), generator=generator).tolist()
    dy = torch.randint(0, 9, (b,), generator=generator).tolist()
    crops = torch.stack(
        [padded[i, :, y : y + 32, x : x + 32] for i, (x, y) in enumerate(zip(dx, dy, strict=True))]
    )
    flip = (torch.rand(b, generator=generator) < 0.5).to(images.device)
    return torch.where(flip.view(-1, 1, 1, 1), crops.flip(-1), crops)


@torch.no_grad()
def test_accuracy(
    model: nn.Module, images: torch.Tensor, labels: torch.Tensor, device: torch.device
) -> float:
    model.eval()
    correct = 0
    for start in range(0, images.shape[0], 1000):
        x = normalize_images(images[start : start + 1000].to(device))
        correct += (model(x).argmax(1).cpu() == labels[start : start + 1000]).sum().item()
    model.train()
    return correct / images.shape[0]


def finetune(
    path: Path,
    images: torch.Tensor,
    labels: torch.Tensor,
    all_images: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    seed: int,
    device: torch.device,
    logger: RunLogger | None = None,
) -> float:
    """Fine-tune the encoder at ``path`` on (images, labels); return test accuracy."""
    seed_everything(seed)
    encoder = load_encoder(path, layer_local=False, device=device)
    features = extract_features(encoder, all_images, device)
    model = Classifier(encoder, features.mean(dim=0), features.std(dim=0)).to(device)
    n = images.shape[0]
    batch = min(MAX_BATCH, n)
    steps = max(MIN_STEPS, math.ceil(EPOCHS * n / batch))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    generator = torch.Generator().manual_seed(seed)
    images, labels = images.to(device), labels.to(device)
    for step in range(steps):
        idx = torch.randint(0, n, (batch,), generator=generator).to(device)
        x = normalize_images(augment(images[idx], generator))
        loss = F.cross_entropy(model(x), labels[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()
        if logger is not None and (step + 1) % 100 == 0:
            logger.log({"train_loss": loss.item()}, step + 1)
    return test_accuracy(model, test_images, test_labels, device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--encoders", nargs="+", required=True, metavar="NAME=PATH")
    parser.add_argument("--seed", type=int, default=0, help="selects the labelled subsets")
    parser.add_argument(
        "--counts", type=int, nargs="*", default=None, help="override finetune.labeled_counts"
    )
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    parser.add_argument(
        "--out", default=None, help="CSV path (default: tables/finetune_s<seed>.csv)"
    )
    args = parser.parse_args()

    counts = args.counts or load_config(args.config)["finetune"]["labeled_counts"]
    out = Path(args.out or REPO_ROOT / "results_and_figs" / "tables" / f"finetune_s{args.seed}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_images, train_labels = load_cifar10("train")
    test_images, test_labels = load_cifar10("test")

    new_file = not out.exists()
    with open(out, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["model", "labeled_n", "seed", "test_acc", "encoder"])
        for spec in args.encoders:
            name, _, path = spec.partition("=")
            path = Path(path) / "encoder.pt" if Path(path).is_dir() else Path(path)
            for n in counts:
                idx = labeled_subset(train_labels, n, args.seed)
                run = f"{name}_n{n}_s{args.seed}"
                log_dir = out_root() / "finetune" / run
                log_dir.mkdir(parents=True, exist_ok=True)
                logger = RunLogger(
                    f"finetune/{name}_n{n}",
                    run,
                    log_dir,
                    {
                        "encoder": str(path),
                        "labeled_n": n,
                        "seed": args.seed,
                        "lr": LR,
                        "weight_decay": WEIGHT_DECAY,
                    },
                )
                acc = finetune(
                    path,
                    train_images[idx],
                    train_labels[idx],
                    train_images,
                    test_images,
                    test_labels,
                    args.seed,
                    device,
                    logger,
                )
                logger.log({"test_acc": acc}, 0)
                logger.close()
                writer.writerow([name, n, args.seed, f"{acc:.4f}", str(path)])
                f.flush()
                print(f"{name} n={n}: test_acc={acc:.4f}", flush=True)
    print(f"results appended to {out}")


if __name__ == "__main__":
    main()
