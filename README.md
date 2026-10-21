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

**Main env.** Versions are pinned in `requirements.txt` after the first working install. Install
torch from the CUDA index that matches the machine (cu118 on the cluster).

```bash
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
.venv/Scripts/python -m pip install -r requirements.txt
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
    --batch_size 64 --num_workers 0 --max_steps 20 --experiment_name smoke
```

This is a 20-step smoke test. `--num_workers 0` avoids slow worker start-up on Windows. CIFAR-10
is expected at `data/upstream_home/data/datasets/cifar10/`; upstream downloads it there if it is
missing.

## Running our pipeline

Not implemented yet. Scripts will run from the repo root with a condition config, for example
`python scripts/pretrain.py --config configs/lpl.yaml`. See [CLAUDE.md](CLAUDE.md).

## Documentation

All documents live in `docs/`. After each milestone, experiment or discovery, add an entry to the
top of [docs/update.md](docs/update.md).

## Licence

`vendor/lpl_zenke/` is MIT-licensed by its authors (see `vendor/lpl_zenke/LICENSE`).
