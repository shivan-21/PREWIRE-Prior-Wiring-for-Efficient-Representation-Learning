# Updates

One entry per milestone, experiment or discovery. **Newest entry on top.** Each entry gives the
date, what was done or found, decisions and why, and what comes next. Link MLflow runs, figures and
commits where they exist.

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

1. Finish the CIFAR-10 download and run a 20-step upstream smoke test.
2. Choose the shared backbone (VGG-11 is the safe default).
3. Write `src/utils/` (config, seed), the CIFAR-10 loader with class-balanced nested labelled
   subsets, and `src/learning_rules/lpl.py` with a parity test against upstream.
