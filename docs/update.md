# Updates

One entry per milestone, experiment or discovery. **Newest entry on top.** Each entry gives the
date, what was done or found, decisions and why, and what comes next. Link MLflow runs, figures and
commits where they exist.

---

## 2026-10-21 — Upstream LPL smoke test passes

**Milestone: upstream LPL runs end to end on this laptop.** Setup is complete.

- CIFAR-10 downloaded to `data/upstream_home/data/datasets/cifar10/` (MD5 verified).
- 60-step run of unmodified `lpl_main.py` (VGG-11, batch 64, otherwise upstream defaults) in
  `.venv-upstream`: trained on the RTX 3060 at about 3 it/s, ran the end-of-epoch linear readout,
  saved a checkpoint and exited cleanly. `vendor/` was left unchanged. Command in the README.
- Per-layer losses at step 59 (logs: `data/upstream_home/data/lpl/cifar10/smoke/`):

  | Layer | Pull | Push | Decorr |
  | --- | --- | --- | --- |
  | 1 | 0.16 | 3.0 | 7e-3 |
  | 2 | 0.022 | 5.4 | 1e-4 |
  | 4 | 4.7e-4 | 8.3 | 4e-8 |
  | 7 | 5.6e-7 | 9.2 | 5e-14 |
  | Final | 1.2e-7 | 9.2 | 2e-15 |

- These numbers only show the plumbing works. Upstream warms the learning rate up from 0 over 10
  epochs, so after 60 steps the network has barely changed. Activation variance shrinks with
  depth: the push loss is −mean log(variance + 1e-4), which tops out at 9.21 when variance is
  zero, so layers 7 and final (9.2) have almost no variance yet. The linear readout is at chance
  (10%). Nothing about LPL's quality can be read from this run.
- Speed: one epoch at batch 64 is about 4 minutes here. Upstream's full schedule (800 epochs,
  batch 1024) is a cluster job, not a laptop one.

**Next.** Choose the backbone, then start on `src/` (utilities, CIFAR-10 loader, LPL
re-implementation with parity test).

---

## 2026-10-21 — Project set up; temporal test redesigned

**Milestone: repository and environments ready.**

- Repository scaffold created (`src/`, `scripts/`, `configs/`, `results_and_figs/`, `docs/`).
- Upstream LPL vendored unmodified at commit `0210f55` (2025-01-14) in `vendor/lpl_zenke/`, with
  only the deep-learning code (no notebooks or spiking simulations).
- Two environments:
  - `.venv-upstream` (Python 3.10, torch 1.11 + cu113, pytorch-lightning 1.6.1) runs upstream
    as-is. Imports work and the RTX 3060 laptop GPU is visible.
  - `.venv` (Python 3.11) for our pipeline. Only ruff is installed so far; torch comes next.
- Upstream's `VGG11Encoder` runs with plain PyTorch (8 layer outputs), so our pipeline can import
  it without Lightning.
- `CLAUDE.md` cleaned up. Epochs, learning rates, batch sizes, seeds and the labelled-count grid
  are no longer fixed in advance; they are chosen from pilot runs and recorded in the configs. The
  backbone is not chosen yet (upstream LPL is validated only on layer-local VGG-11).

**Discovery: CIFAR-10 has no temporal structure.**

- In upstream LPL on CIFAR-10, the two "consecutive time steps" are two random augmentations of
  the same image (crop, flip, colour jitter, greyscale). They stand in for views of one object,
  not for frames in time.
- LPL and SimCLR therefore train on the same view pairs. They differ in the loss (predictive +
  Hebbian + decorrelation vs contrastive) and in layer-local vs end-to-end training.
- Upstream's `--shuffle_positives` pairs each image's view with another image's view. On CIFAR-10
  this removes the same-object pairing; it does not test temporal order.
- Upstream has real sequences only for Shapes3D (synthetic, 4 shape classes).

**Decision: keep CIFAR-10 and build synthetic sequences from it.** No other dataset for now.

- Main labelled-data comparison (random, Oja, LPL, SimCLR): CIFAR-10 with standard two-view pairs.
- Temporal test (`lpl_temporal`): each CIFAR-10 image becomes a short sequence whose crop,
  rotation and colour change smoothly; LPL compares frames a set number of steps apart.
  - `lpl_temporal` vs `lpl` isolates temporal smoothness (frames should use roughly the same
    augmentation distribution as the two-view pairs).
  - Shuffled positives on sequences remove the same-object pairing.
  - Sweeping the frame gap: large gaps should behave like independent views.
- This replaces "scramble the temporal order of LPL's training stream" in the working summary.

**Next.**

1. Finish the CIFAR-10 download and run an upstream smoke test (done, see entry above).
2. Choose the shared backbone (VGG-11 is the safe default).
3. Write `src/utils/` (config, seed), the CIFAR-10 loader with class-balanced nested labelled
   subsets, and `src/learning_rules/lpl.py` with a parity test against upstream.
