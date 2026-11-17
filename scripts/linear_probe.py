"""Linear probes on frozen features of every layer, for each labelled-set size.

    python scripts/linear_probe.py --encoders random=results_and_figs/checkpoints/random_init_s0 \
        lpl=results_and_figs/checkpoints/lpl_s0 --seed 0

Each ``name=path`` points at a pre-training output directory (its ``encoder.pt``) or a ``.pt`` file.
For every encoder, layer and labelled-set size in ``finetune.labeled_counts`` (from
``configs/base.yaml``), a logistic regression is fit on the class-balanced labelled subset
(``src/data/cifar.labeled_subset``) and scored on the 10k test images. Features are standardised
with statistics of all 50k unlabelled training images. Results are appended to
``results_and_figs/tables/linear_probe_s<seed>.csv`` and logged to MLflow (``linear_probe/<name>``).
"""

import argparse
import csv
from pathlib import Path

import torch

from src.analysis.linear_probe import extract_layer_features, linear_probe_accuracy
from src.data.cifar import labeled_subset, load_cifar10
from src.models.vgg import load_encoder
from src.utils.config import load_config
from src.utils.logging import RunLogger
from src.utils.paths import REPO_ROOT, out_root


def encoder_file(path: str) -> Path:
    path = Path(path)
    return path / "encoder.pt" if path.is_dir() else path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--encoders", nargs="+", required=True, metavar="NAME=PATH")
    parser.add_argument("--seed", type=int, default=0, help="selects the labelled subsets")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    parser.add_argument(
        "--out", default=None, help="CSV path (default: tables/linear_probe_s<seed>.csv)"
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", default=None, help="probe only these layers (1-8)"
    )
    args = parser.parse_args()

    counts = load_config(args.config)["finetune"]["labeled_counts"]
    out = Path(
        args.out or REPO_ROOT / "results_and_figs" / "tables" / f"linear_probe_s{args.seed}.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_images, train_labels = load_cifar10("train")
    test_images, test_labels = load_cifar10("test")
    subsets = {n: labeled_subset(train_labels, n, args.seed) for n in counts}

    new_file = not out.exists()
    with open(out, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["model", "layer", "labeled_n", "seed", "test_acc", "encoder"])
        for spec in args.encoders:
            name, _, path = spec.partition("=")
            path = encoder_file(path)
            encoder = load_encoder(path, layer_local=True, device=device)
            train_feats = extract_layer_features(encoder, train_images, device)
            test_feats = extract_layer_features(encoder, test_images, device)
            log_dir = out_root() / "linear_probe" / f"{name}_s{args.seed}"
            log_dir.mkdir(parents=True, exist_ok=True)
            logger = RunLogger(
                f"linear_probe/{name}",
                f"{name}_s{args.seed}",
                log_dir,
                {"encoder": str(path), "seed": args.seed, "labeled_counts": counts},
            )
            for layer, (tr, te) in enumerate(zip(train_feats, test_feats, strict=True), start=1):
                if args.layers and layer not in args.layers:
                    continue
                stats = (tr.mean(dim=0), tr.std(dim=0))
                for n, idx in subsets.items():
                    idx = idx.to(device)
                    acc = linear_probe_accuracy(
                        tr[idx], train_labels[idx.cpu()].to(device), te, test_labels, stats=stats
                    )
                    writer.writerow([name, layer, n, args.seed, f"{acc:.4f}", str(path)])
                    logger.log({f"acc_layer{layer}_n{n}": acc}, step=0)
                    print(f"{name} layer {layer} n={n}: {acc:.4f}", flush=True)
            f.flush()
            logger.close()
    print(f"results appended to {out}")


if __name__ == "__main__":
    main()
