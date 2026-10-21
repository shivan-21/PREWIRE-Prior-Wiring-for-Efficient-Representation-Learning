# Updates

One entry per milestone, experiment or discovery. **Newest entry on top.** Each entry gives the
date, what was done or found, decisions and why, and what comes next. Link MLflow runs, figures and
commits where they exist.

---

## 2026-10-21 — Temporal stream designed and built

**Decisions.**

- **Backbone: VGG-11**, upstream's layer-local `VGG11Encoder`, for every condition.
- **Augmentations: exactly upstream's SimCLR set**, made smooth over time: crop (scale 0.08–1,
  ratio 3/4–4/3), flip, colour jitter (0.8/0.8/0.8/0.2, p 0.8, random order), greyscale (p 0.2),
  3×3 blur (σ 0.1–2, p 0.5). No rotation or other extras, so the temporal condition differs from
  `lpl` only in how frames relate over time, not in which augmentations are seen.
- **Clip generation: Gaussian-copula AR(1).** Each augmentation parameter is driven by a latent
  with frame-to-frame correlation ρ, mapped to upstream's distribution. Every frame alone has
  exactly upstream's distribution; ρ = 0 gives upstream's independent views, so `lpl`, `simclr`
  and `lpl_temporal` share one augmentation implementation.
- **Budget: equal frames processed.** 2048 frames per step in every condition (1024 images × 2
  views, or 256 clips × 8 frames) for the same number of steps; schedules defined in steps.
- **Starting values (pilot defaults):** clip length 8, ρ = 0.8, frame gap 1. Optimiser and loss
  settings stay at upstream's (Adam 1e-3, wd 1.5e-6, pull/push/decorr 1/1/10). The step budget
  is set later from an `lpl` pilot with linear probes (upstream's full schedule is ~35k steps).
- **Temporal controls at the same budget:** ρ = 0 clips (main test of smoothness), frames
  shuffled within each clip (order vs similarity), cross-clip shuffled positives (same-image
  pairing), and a ρ sweep (0.5 / 0.8 / 0.95).

**Built** (branch `data/temporal-clips`): `src/data/augmentations.py`, `src/data/sequences.py`,
`src/data/cifar.py`, `src/utils/paths.py`, `configs/lpl_temporal.yaml`,
`scripts/preview_clips.py`, `tests/test_augmentations.py` (23 tests). Main env installed
(torch 2.7.1+cu118) and `requirements.txt` pinned.

**Checks.**

- Per-parameter distributions match upstream for ρ ∈ {0, 0.8, 0.95} (KS, binomial and χ² tests),
  and lag correlations equal ρ^Δ. Colour ops, greyscale, blur and crop resampling match
  torchvision's own functions.
- Crop sizes match torchvision's rejection sampling, including its rounding: mean crop area
  0.4896 vs 0.4894.
- Against real upstream frames (`SimCLRTrainDataTransform`, 10k images), our ρ = 0 frames have
  matching per-channel means, standard deviations, sharpness and colourfulness (KS distance ≤
  0.019). The small remaining offset (≤ 0.01 in normalised units) is consistent with upstream's
  8-bit rounding.
- GPU and CPU outputs agree (max difference 1.6e-5). Generating one 2048-frame batch takes
  78 ms on the laptop GPU.

**Findings from the clip preview** (`results_and_figs/figures/temporal_clips_preview_20261021.pdf`).

- A first version mapped the jitter order to 24 bins of one latent, so the order changed almost
  every frame even at ρ = 0.95. It now comes from the ranking of four latents: at ρ = 0.95 it
  changes in 46% of frames, and a swap changes a frame only ~1% as much as normal drift.
- On/off switches still happen at high ρ (each in about 7–10% of frames at ρ = 0.95, about 20%
  at ρ = 0.8). They are part of upstream's distribution, so they stay.
- Consecutive-frame difference (pixel MSE) falls from 0.123 at ρ = 0 to 0.057 at ρ = 0.8 and
  0.027 at ρ = 0.95.
- At ρ = 0.95, an 8-frame clip covers little of the augmentation range (first and last frames
  still have latent correlation 0.70). Pilots should check that positives aren't too similar for
  the pull loss to teach anything; this is the main reason the ρ sweep exists.

**Next.** Re-implement the LPL loss in `src/learning_rules/lpl.py` with a parity test against
upstream's `module.py`, then a pre-training script, then laptop pilots of `lpl` vs `lpl_temporal`.

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
