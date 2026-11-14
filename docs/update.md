# Updates

One entry per milestone, experiment or discovery. **Newest entry on top.** Each entry gives the
date, what was done or found, decisions and why, and what comes next. Link MLflow runs, figures and
commits where they exist.

---

## 2026-11-14 — Remaining runs done: temporal test and seeds 1–2 copied to the laptop

**Queued 2026-11-11** with `cluster/queue_remaining.sh` (three afterany lanes, jobs
2644625–2644650), after laptop pilots of every temporal variant (rho 0.8 for 400 steps: loss
17.8 → 12.3, push 0.43–2.35 per layer; the others 60 steps each). All 13 runs finished (exit 0).
Temporal runs took ~0.29 s/step on the V100, about twice standard LPL.

**Copied to the laptop** (`results_and_figs/checkpoints/<run>/`, plus `mlruns/` and
`cluster_logs/`). The first bulk transfer broke mid-stream; a per-run copy-and-verify loop then
matched every run's file count and byte total to the cluster.

**Online probe at the end of training** (last layer, 10k train / 10k test images; a monitoring
number, not the evaluation protocol):

| Run | Final probe |
| --- | --- |
| LPL seeds 0 / 1 / 2 | 60.9 / 60.5 / 61.0% |
| SimCLR seeds 0 / 1 / 2 | 79.4 / 79.4 / 79.5% |
| Oja seeds 0 / 1 / 2 | 49.5 / 50.1 / 51.0% |
| LPL, two views paired across images (`lpl_shuffled`) | 39.4% |
| LPL on clips, rho 0.8 (`lpl_temporal`) | 65.2% |
| — rho 0 (independent frames, same clip layout) | 62.4% |
| — frames shuffled within each clip | 64.7% |
| — frames paired across clips | 39.0% |
| — rho 0.5 / rho 0.95 | 64.7 / 59.5% |

First impressions only (single seed for the temporal runs; the full evaluation comes next):
seed-to-seed spread is under ~1.5 points; pairing views of *different* images destroys LPL's
benefit (below random init's ~42.5%); smooth clips beat independent frames by ~3 points, but
shuffling the clip order costs only ~0.5 points; very smooth clips (rho 0.95) are worse.

**Next (M4).** Agree the analysis plan, then run linear probes, fine-tuning and figures on the
laptop.

---

## 2026-11-11 — M3: pre-training done, results copied to the laptop

**Runs** (AQUA, single seed, 35,000 steps × 2048 frames each; all exit 0 on 2026-11-04):

| Run | Job | Online probe (last layer, 10k/10k) |
| --- | --- | --- |
| LPL | 2576677 | 42.5% at step 0 → **60.9%** at 35k (max 61.4%) |
| SimCLR | 2576700 | 42.4% → **79.4%** (max 79.5%) |
| Oja | 2576681 | 42.5% → **49.5%** (max 50.7%) |

**Upstream reference** (unmodified `lpl_main.py`, legacy env on AQUA): the 2-epoch smoke job passed;
the full 800-epoch run hit the 48 h walltime at **epoch 656** (~4.4 min/epoch; upstream loads and
augments data on the CPU). Its last checkpoint (`epoch=655-step=26240.ckpt`) loads into our
VGG-11, so it can be probed like our encoders.

**Copied to the laptop** (`results_and_figs/checkpoints/`, gitignored): `lpl_s0/`, `simclr_s0/`,
`oja_s0/` (encoder, full state, metrics, 14 snapshots each), the cluster `mlruns/`, the upstream
reference run and checkpoint, calibration metrics, and all job logs (`cluster_logs/`). File counts
and byte totals match the cluster exactly (e.g. `lpl_s0`: 17 files, 664,766,460 bytes).

**Next (M4).** Agree the analysis, then run the linear probes and fine-tuning on the laptop.

---

## 2026-11-04 — M2: cluster set up and calibrated; main runs submitted

**Agreed at the M1 check-in.** Run upstream's own code on the cluster as a reference for our LPL
(drop it if the legacy stack won't run there), and report probe accuracy at every layer as a
second figure. Go-ahead for M2 and for starting the cluster runs.

**Set-up on AQUA.**

- Repository cloned via a read-only deploy key (`aqua-prewire`). GitHub intermittently rejects
  the key or returns 500 errors; retrying `git pull` works.
- CIFAR-10 uploaded from the laptop (MD5 verified) to `~/scratch/prewire/cifar10/`; the direct
  download from Toronto ran at ~65 KB/s.
- Main env `~/envs/prewire`: torch 2.6.0+cu118 (same as the laptop). The other pins in
  `requirements.txt` have no builds for CentOS 7, so the cluster has the newest that do
  (numpy 2.2.6, scipy 1.16.3, mlflow-skinny 3.18.0; full list in
  `~/prewire/logs/cluster_main_env_freeze.txt`). Only torch affects results.
- Installing onto Lustre is slow (≈2 h for the main env), mostly creating small files.

**Calibration** (300 steps, 2048 frames per step, V100):

| Method | s/step | GPU memory | 35k steps |
| --- | --- | --- | --- |
| LPL | 0.135 | 3.8 GB | ~1.3 h |
| SimCLR | 0.161 | 4.1 GB | ~1.6 h |
| Oja | 0.093 | 2.8 GB | ~0.9 h |

**Budget: 35,000 steps for every condition** = upstream's full schedule (800 epochs × 44 steps of
1024 images). It fits easily, so no trade-off was needed. Upstream's reference can therefore run
its published 800-epoch schedule and be compared directly with our main LPL run (instead of the
short matched pair planned earlier).

**SimCLR collapse in calibration, explained.** The 300-step calibration has a 4-step warmup; at
1024 pairs SimCLR collapsed (loss pinned at log(2N−1) = 7.624). On the laptop with the real
437-step warmup at the same batch size it learns (loss 7.15 → 6.59, probe 47.5% after 600 steps),
so the main run keeps the shared settings. The online probe will show any later collapse.

**Also built (laptop):** evaluation (`scripts/linear_probe.py`, `scripts/finetune.py`), the
data-equivalence calculation, and figures (`scripts/plot_results.py`), tested end to end on the
pilot encoders. The labelled-set grid is now 12 points (10 … 50,000) for a smoother
data-equivalence interpolation; evaluation runs on the laptop in minutes per encoder.

**Running.** Main pre-training jobs for LPL, SimCLR and Oja (single seed, 2-link chains). The
upstream reference follows once its legacy env is installed: a 2-epoch smoke job first, then the
full 800-epoch run.

---

## 2026-11-04 — M1: pre-training code ready

**Plan agreed today.** Results by 2026-11-12 (presentation 2026-11-20). One seed each for the
three pre-trained models (LPL, SimCLR, Oja) plus random init; the temporal test only if time
remains. Defaults approved: Oja = Sanger's rule per conv layer, greedy; SimCLR = NT-Xent,
temperature 0.5, same budget; evaluation = linear probe and full fine-tuning at
10 / 100 / 1k / 10k / 50k labels. Milestones: M1 code (this entry), M2 cluster calibration and
budget, M3 pre-training runs, M4 labelled-data results, M5 wrap-up.

**Built** (branch `train/pretraining-pipeline`):

- `src/learning_rules/lpl.py`: LPL loss re-implemented from upstream `module.py`.
  **Parity test passes** (`tests/test_lpl.py` runs upstream in `.venv-upstream`): per-layer pull,
  push and decorrelation losses and the first/last conv gradients match upstream, with and
  without shuffled positives.
- `src/learning_rules/simclr.py` (NT-Xent + 512→2048→128 head) and `src/learning_rules/oja.py`
  (Sanger's rule on 3×3 patches, greedy per layer, filters used with both signs, unit-variance
  gain). Sanger's rule recovers the true principal components in a unit test.
- `scripts/pretrain.py`: equal frames per step, warmup + cosine schedule in steps, atomic
  checkpoints with automatic resume, MLflow + `metrics.csv`, online linear probe.
- `src/models/vgg.py` (upstream VGG-11 loaded by file path), configs for every condition,
  `cluster/` PBS scripts. 41 tests pass.

**Decisions and findings.**

- **He (Kaiming) init for every condition.** With PyTorch's default init (upstream's), the
  backbone output has std 0.006: SimCLR's loss sat exactly at the collapse value log(2N−1)
  and did not learn. With He init (output std 0.82) it learns. LPL now also starts from He init.
- **torch 2.6.0+cu118** on laptop and cluster: the version proven on AQUA (CentOS 7).
- **Oja step size 0.2**, calibrated on real layer-2 patches against exact PCA: in 3,000 steps
  the top 8/32/64 components capture 0.438/0.640/0.774 of the variance (exact PCA:
  0.441/0.623/0.763). The initial 0.05 under-converged (0.412/0.546/0.660).
- **Resume works**: a run killed after its checkpoint and restarted matches an uninterrupted run
  as closely as two uninterrupted runs match each other (cuDNN is not bit-deterministic).
- MLflow 3.17 refuses folder stores by default; we opt in rather than share an SQLite file
  across cluster jobs on Lustre.

**Laptop pilots** (3,000 steps × 1024 frames, ≈ 25 epochs; full runs will be ~20× longer).
Linear probe, 10k train / 10k test images, per layer:

| Model | Last layer (L8) | Best layer | Online probe trend |
| --- | --- | --- | --- |
| Random (He init) | 43.8% | L3 52.1% | — |
| LPL | 43.3% | L3 52.7% (L4–L7 +1–3 pts over random) | dips to 39.6%, back to 43.4% |
| SimCLR | 53.5% | L5 58.8% | rising steadily, 42% → 53% |
| Oja | 41.4% | L3 54.9% | 45.9% after layer 1, falls as deeper layers train |

- LPL trains stably (push 0.47–1.15 per layer, far from the 9.21 collapse ceiling) but slowly,
  consistent with upstream's 800-epoch schedule. At this budget it barely beats random.
- Every method's last layer is a weaker readout than its middle layers.
- Speed on the laptop (1024 frames/step): LPL 0.17 s, SimCLR 0.23 s, Oja 0.12–0.16 s.

**Open points for the M1 check-in.**

1. Whether to run upstream's own code on the cluster as a reference for our LPL.
2. Whether to report per-layer probes alongside the last-layer probe.

**Next (M2).** Deploy key, clone and environment on AQUA, CIFAR-10 copy, a short calibration
job, then the step budget.

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
