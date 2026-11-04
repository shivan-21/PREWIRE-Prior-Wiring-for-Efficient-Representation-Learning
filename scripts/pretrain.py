"""Phase 1: unsupervised pre-training of the shared VGG-11 (LPL, SimCLR, Oja or random init).

    python scripts/pretrain.py --config configs/lpl.yaml
    python scripts/pretrain.py --config configs/lpl.yaml --set pretrain.max_steps=200 --run_name s

Outputs go to ``$PREWIRE_OUT/<run_name>/`` (default ``results_and_figs/checkpoints/<run_name>/``):
``state.pt`` (full training state, rewritten every ``ckpt_every`` steps), ``metrics.csv``,
``snapshots/step_*.pt`` (encoder only, every ``snapshot_every`` steps) and, when
finished, ``encoder.pt``. A run whose ``state.pt`` exists resumes from it; a run whose
``encoder.pt`` exists exits immediately, so chained cluster jobs are safe.

Every step processes ``frames_per_step`` frames: ``frames_per_step / clip_length`` CIFAR-10 training
images, each turned into a clip by ``src/data/sequences.make_clips``. Data order and augmentation
randomness are functions of (seed, step), so a resumed run sees exactly the batches it would have.
"""

import argparse
import time

import torch

from src.analysis.linear_probe import extract_features, linear_probe_accuracy
from src.data.cifar import load_cifar10
from src.data.sequences import frame_pairs, make_clips
from src.learning_rules.lpl import LPLCoefficients, lpl_loss
from src.learning_rules.oja import OjaPretrainer
from src.learning_rules.simclr import ProjectionHead, nt_xent_loss
from src.models.vgg import vgg11
from src.utils.checkpoint import load_if_exists, save_atomic
from src.utils.config import load_config
from src.utils.logging import RunLogger
from src.utils.paths import out_root
from src.utils.schedule import warmup_cosine_lr
from src.utils.seed import seed_everything, step_generator

METHODS = ("random", "lpl", "simclr", "oja")
PROBE_TRAIN_SIZE = 10_000


def batch_indices(step: int, images_per_step: int, num_images: int, seed: int) -> torch.Tensor:
    """Training-image indices for a step.

    A fresh permutation each epoch; the last partial batch of an epoch is dropped.
    """
    steps_per_epoch = num_images // images_per_step
    epoch, position = divmod(step, steps_per_epoch)
    perm = torch.randperm(num_images, generator=torch.Generator().manual_seed(seed * 7919 + epoch))
    return perm[position * images_per_step : (position + 1) * images_per_step]


def param_groups(modules: list[torch.nn.Module], weight_decay: float) -> list[dict]:
    """Weight decay on weights only, not biases (as upstream)."""
    decay, no_decay = [], []
    for module in modules:
        for name, param in module.named_parameters():
            (no_decay if "bias" in name else decay).append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="SECTION.KEY=VALUE", help="config overrides"
    )
    parser.add_argument("--run_name", default=None, help="default: <config name>_s<seed>")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    p, seq, seed = cfg["pretrain"], cfg["data"]["sequence"], cfg["seed"]
    method = p["method"]
    if method not in METHODS:
        raise ValueError(f"pretrain.method must be one of {METHODS}, got {method!r}")
    run_name = args.run_name or f"{cfg['name']}_s{seed}"
    out_dir = out_root() / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder_path, state_path = out_dir / "encoder.pt", out_dir / "state.pt"
    if encoder_path.exists():
        print(f"{run_name}: already finished ({encoder_path}), nothing to do")
        return

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = vgg11(layer_local=method in ("lpl", "oja")).to(device)

    if method == "random":
        save_atomic({"encoder": encoder.state_dict(), "config": cfg, "step": 0}, encoder_path)
        print(f"{run_name}: saved random init to {encoder_path}")
        return

    max_steps = p["max_steps"]
    if not max_steps:
        raise ValueError("set pretrain.max_steps (in the config or with --set)")
    clip_length = 1 if method == "oja" else seq["clip_length"]
    images_per_step = p["frames_per_step"] // clip_length
    warmup_steps = round(p["warmup_fraction"] * max_steps)

    images, labels = load_cifar10("train")
    images = images.to(device)
    test_images, test_labels = load_cifar10("test")
    probe_images, probe_labels = images[:PROBE_TRAIN_SIZE], labels[:PROBE_TRAIN_SIZE]

    head = ProjectionHead().to(device) if method == "simclr" else None
    optimizer, oja = None, None
    if method == "oja":
        oja = OjaPretrainer(
            encoder, max_steps, p["oja"]["lr"], p["oja"]["patches_per_step"], seed, device
        )
    else:
        modules = [encoder] + ([head] if head is not None else [])
        optimizer = torch.optim.Adam(param_groups(modules, p["weight_decay"]), lr=p["lr"])
    coefficients = LPLCoefficients(**p.get("lpl_coefficients", {}))

    start_step, run_id = 0, None
    state = load_if_exists(state_path, device)
    if state is not None:
        encoder.load_state_dict(state["encoder"])
        if head is not None:
            head.load_state_dict(state["head"])
        if optimizer is not None:
            optimizer.load_state_dict(state["optimizer"])
        if oja is not None:
            oja.load_state_dict(state["oja"])
        start_step, run_id = state["step"], state["run_id"]
        print(f"{run_name}: resuming from step {start_step}")

    logger = RunLogger(p.get("experiment") or f"pretrain/{method}", run_name, out_dir, cfg, run_id)
    logger.set_tags({"method": method, "out_dir": str(out_dir)})

    def save_state(step: int) -> None:
        save_atomic(
            {
                "step": step,
                "run_id": logger.run_id,
                "config": cfg,
                "encoder": encoder.state_dict(),
                "head": None if head is None else head.state_dict(),
                "optimizer": None if optimizer is None else optimizer.state_dict(),
                "oja": None if oja is None else oja.state_dict(),
            },
            state_path,
        )

    def probe() -> float:
        train_x = extract_features(encoder, probe_images, device)
        test_x = extract_features(encoder, test_images, device)
        return linear_probe_accuracy(train_x, probe_labels.to(device), test_x, test_labels)

    if start_step == 0:
        logger.log({"probe_acc": probe()}, 0)

    tick = time.time()
    for step in range(start_step, max_steps):
        idx = batch_indices(step, images_per_step, images.shape[0], seed)
        frames = make_clips(
            images[idx.to(device)],
            clip_length,
            seq["rho"],
            seq["order"],
            generator=step_generator(seed, step),
        )
        num_clips = frames.shape[0]
        log_now = (step + 1) % p["log_every"] == 0
        metrics: dict[str, float] = {}

        if method == "oja":
            metrics = oja.step(frames[:, 0], step)
        else:
            lr = warmup_cosine_lr(
                step, max_steps, p["lr"], warmup_steps, p["start_lr"], p["final_lr"]
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            flat = frames.flatten(0, 1)
            if method == "lpl":
                _, _, z = encoder(flat)
                pairs = [
                    frame_pairs(zl.view(num_clips, clip_length, -1), seq["frame_gap"]) for zl in z
                ]
                z1, z2 = [a for a, _ in pairs], [b for _, b in pairs]
                if p["shuffle_positives"]:
                    perm = torch.randperm(z2[0].shape[0], generator=step_generator(seed, step, 1))
                    z2 = [b[perm.to(device)] for b in z2]
                loss, terms = lpl_loss(z1, z2, coefficients)
            else:
                pooled, _, _ = encoder(flat)
                h = head(pooled).view(num_clips, clip_length, -1)
                h1, h2 = frame_pairs(h, seq["frame_gap"])
                loss = nt_xent_loss(h1, h2, p["simclr"]["temperature"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if log_now:
                metrics = {"loss": loss.item(), "lr": lr}
                if method == "lpl":
                    for name, values in terms.items():
                        metrics.update(
                            {f"{name}_layer{i + 1}": v.item() for i, v in enumerate(values)}
                        )

        if log_now:
            if device.type == "cuda":
                torch.cuda.synchronize()
            metrics["sec_per_step"] = (time.time() - tick) / p["log_every"]
            if device.type == "cuda":
                metrics["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
            logger.log(metrics, step + 1)
            print(
                f"step {step + 1}/{max_steps} "
                + " ".join(f"{k}={v:.4g}" for k, v in metrics.items() if "_layer" not in k),
                flush=True,
            )
            tick = time.time()
        if (step + 1) % p["probe_every"] == 0 or step + 1 == max_steps:
            acc = probe()
            logger.log({"probe_acc": acc}, step + 1)
            print(f"step {step + 1}: probe_acc={acc:.4f}", flush=True)
            tick = time.time()
        if p.get("snapshot_every") and (step + 1) % p["snapshot_every"] == 0:
            save_atomic(
                {"encoder": encoder.state_dict(), "config": cfg, "step": step + 1},
                out_dir / "snapshots" / f"step_{step + 1:06d}.pt",
            )
        if (step + 1) % p["ckpt_every"] == 0:
            save_state(step + 1)

    save_state(max_steps)
    save_atomic({"encoder": encoder.state_dict(), "config": cfg, "step": max_steps}, encoder_path)
    logger.set_tags({"encoder_path": str(encoder_path)})
    logger.close()
    print(f"{run_name}: finished, encoder saved to {encoder_path}")


if __name__ == "__main__":
    main()
