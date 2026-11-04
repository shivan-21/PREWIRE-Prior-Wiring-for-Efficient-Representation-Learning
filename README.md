# PREWIRE: Prior Wiring for Efficient Representation Learning

How much labelled data does a brain-like prior save? We pre-train a network without labels using
Latent Predictive Learning (LPL; Halvagal & Zenke, 2023, *Nature Neuroscience*) and compare it with
random initialisation, Oja's rule and SimCLR. Each pre-trained network is then fine-tuned on
CIFAR-10 with 10 to 50,000 labels. A second experiment trains LPL on synthetic sequences built from
CIFAR-10 images to test whether temporal structure is what makes LPL work.

CS6251 Computational Models of Cognition, IIT Madras (Prof. Sutanu Chakraborti).
Shivan Ajay Iyer and Aditya Suhas Tamras.

- Project conventions and design decisions: [CLAUDE.md](CLAUDE.md)
- Progress, results and decisions, newest first: [docs/update.md](docs/update.md)
- Proposal and working summary: [docs/](docs/)

## Repository layout

| Path | Contents |
| --- | --- |
| `src/` | Importable package: models, data loaders, learning rules, analysis |
| `scripts/` | Entry points (pre-train, fine-tune, evaluate), run from the repo root |
| `configs/` | YAML configs: `base.yaml` plus one file per condition |
| `vendor/lpl_zenke/` | Upstream LPL code, unmodified; `VERSION.txt` records the commit |
| `results_and_figs/` | Figures, tables and checkpoints (checkpoints are gitignored) |
| `docs/` | All documents: update log, proposal, reports, slides |
| `data/` | Datasets (gitignored) |

## Setup

There are two Python environments, because upstream LPL needs an old library stack (torch 1.11,
pytorch-lightning 1.6) that does not install on Python 3.11.

| Env | Python | Used for |
| --- | --- | --- |
| `.venv` | 3.11 | Our pipeline (`src/`, `scripts/`) |
| `.venv-upstream` | 3.10 | Running upstream LPL as-is, and the parity test for our LPL re-implementation |

Commands below are for Windows (PowerShell or Git Bash), run from the repo root.

**Main env.** Versions are pinned in `requirements.txt`. torch comes from the CUDA 11.8 index
(laptop and cluster). The editable install makes `src` importable from `scripts/` and `tests/`.

```bash
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pip install -e .
.venv/Scripts/pytest            # augmentation and sequence tests
```

**Legacy env (upstream LPL).** pytorch-lightning 1.6.1 has package metadata that pip 24.1 and
later reject, so pip is held below 24.1.

```bash
py -3.10 -m venv .venv-upstream
.venv-upstream/Scripts/python -m pip install "pip<24.1"
.venv-upstream/Scripts/python -m pip install -r vendor/lpl_zenke/requirements-legacy.txt \
    --extra-index-url https://download.pytorch.org/whl/cu113
```

**Low disk space on the system drive.** pip unpacks torch (about 2 GB) into the temp folder and
caches it. If the system drive is nearly full, keep the repo on a larger drive and point pip there:

```bash
TMP="$PWD/.venv-upstream/piptmp" TEMP="$PWD/.venv-upstream/piptmp" \
    .venv-upstream/Scripts/python -m pip install --no-cache-dir -r ...
```

## Running upstream LPL

Upstream uses top-level imports (`from models.encoders import ...`), so it runs from inside
`vendor/lpl_zenke/`. It reads and writes `~/data/datasets` and `~/data/lpl`. To keep those inside
the repo's `data/` folder, point the home directory at `data/upstream_home` (on Windows, Python
reads `USERPROFILE`):

```bash
cd vendor/lpl_zenke
USERPROFILE="$(cd ../.. && pwd)/data/upstream_home" HOME="$(cd ../.. && pwd)/data/upstream_home" \
    ../../.venv-upstream/Scripts/python lpl_main.py \
    --batch_size 64 --num_workers 0 --max_steps 60 --experiment_name smoke
```

This is a 60-step smoke test (about 20 s of training on an RTX 3060 laptop GPU, plus a short
linear-readout evaluation at the end).

- `--max_steps 60`: Lightning logs every 50 steps, so shorter runs record no per-layer losses.
- `--num_workers 0`: avoids slow worker start-up on Windows.
- Don't add `--verbose_printing` to a run that stops mid-epoch: its table printer crashes
  (`KeyError: 'Layerwise validation losses/...'`) because no validation metrics exist yet.
- Upstream refuses to overwrite an existing experiment. Delete
  `data/upstream_home/data/lpl/cifar10/smoke/` or change `--experiment_name` before re-running.
- Outputs (TensorBoard log, checkpoint) go to `data/upstream_home/data/lpl/cifar10/<experiment_name>/`.

CIFAR-10 is expected at `data/upstream_home/data/datasets/cifar10/`; upstream downloads it there if
it is missing. The download URL has moved to `https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz`
(MD5 `c58f30108f718f92721af3b95e74349a`). The server can't resume interrupted downloads.

## Running our pipeline

Scripts run from the repo root with a condition config (`configs/*.yaml`, merged over
`configs/base.yaml`). See [CLAUDE.md](CLAUDE.md) for the conditions.

**Pre-training** (`lpl`, `simclr`, `oja`, `random_init`, `lpl_shuffled`, `lpl_temporal`):

```bash
python scripts/pretrain.py --config configs/lpl.yaml
python scripts/pretrain.py --config configs/lpl.yaml --run_name pilot_lpl     --set pretrain.max_steps=3000 pretrain.frames_per_step=1024   # overrides
```

- Outputs go to `$PREWIRE_OUT/<run_name>/` (default `results_and_figs/checkpoints/<run_name>/`):
  `state.pt` (full training state, every `ckpt_every` steps), `metrics.csv`, and `encoder.pt`
  when finished. Re-running resumes from `state.pt`; a finished run exits immediately.
- Metrics also go to MLflow (`./mlruns` unless `MLFLOW_TRACKING_URI` is set):
  `mlflow ui --backend-store-uri ./mlruns` with `MLFLOW_ALLOW_FILE_STORE=true`.
- An online linear probe (10k train / 10k test images, frozen last-layer features) is logged
  every `probe_every` steps as `probe_acc`.
- On the laptop GPU (6 GB), 1024 frames per step takes ~0.17 s (LPL) to ~0.23 s (SimCLR).

**Cluster.** `cluster/submit.sh` submits a chain of PBS jobs for one config (see `CLUSTER.md`,
kept out of git), e.g. `bash cluster/submit.sh configs/lpl.yaml`.

**Other tools.**

- `python scripts/preview_clips.py --rhos 0 0.5 0.8 0.95` renders example synthetic clips to
  `results_and_figs/figures/`.
- CIFAR-10 is read from `$PREWIRE_DATA/cifar10/cifar-10-batches-py`. Without `PREWIRE_DATA` it
  defaults to `data/upstream_home/data/datasets/`, the copy upstream also uses.

## Documentation

All documents live in `docs/`. After each milestone, experiment or discovery, add an entry to the
top of [docs/update.md](docs/update.md).

## Licence

`vendor/lpl_zenke/` is MIT-licensed by its authors (see `vendor/lpl_zenke/LICENSE`).
