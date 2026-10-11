# CLAUDE.md — LPL Data Efficiency Project

## Project Overview

This is a research project for CS6251 (Computational Models of Cognition, IIT Madras, Prof. Sutanu Chakraborti). We investigate the **knowledge–data tradeoff** by measuring how much labeled data a biologically plausible learning rule (Latent Predictive Learning, LPL) saves compared to other initialization strategies.

**Core question:** Does LPL's temporal predictive prior reduce labeled data requirements, and is temporal structure the active ingredient?

**Key paper:** Halvagal & Zenke (2023), *Nature Neuroscience*, 26(11), 1906–1915.

## Repository Structure

Modules are created when they are implemented; the tree below is the intended layout.

```
PREWIRE/
├── CLAUDE.md                  # This file
├── CLUSTER.md                 # HPC cluster instructions (gitignored, local only)
├── pyproject.toml             # Ruff config (excludes vendor/)
├── requirements.txt           # Main env (Python 3.11)
├── configs/                   # YAML experiment configs
│   ├── base.yaml              # Shared defaults
│   ├── random_init.yaml       # Random initialization condition
│   ├── oja.yaml               # Oja's rule pre-training
│   ├── lpl.yaml               # LPL pre-training
│   ├── simclr.yaml            # SimCLR pre-training
│   └── lpl_shuffled.yaml      # LPL with shuffled positive pairs (ablation)
├── src/                       # Importable Python package
│   ├── models/                # Shared backbone + classification / linear-probe heads
│   ├── data/
│   │   ├── cifar.py           # CIFAR-10 loaders with labeled-subset sampling
│   │   └── augmentations.py   # Two-view augmentation pipeline (LPL and SimCLR)
│   ├── learning_rules/
│   │   ├── oja.py             # Oja's normalized Hebbian rule
│   │   ├── lpl.py             # LPL, built on vendor/lpl_zenke
│   │   └── simclr.py          # SimCLR baseline
│   ├── analysis/
│   │   ├── linear_probe.py    # Frozen-feature linear probes
│   │   ├── data_equivalence.py# Data-equivalence ratio ("LPL's prior is worth N labels")
│   │   ├── cka.py             # CKA representational similarity (exploratory)
│   │   └── fca.py             # FCA analysis of activations (exploratory)
│   └── utils/
│       ├── config.py          # YAML config loader
│       ├── logging.py         # MLflow integration helpers
│       └── seed.py            # Reproducibility (seed_everything)
├── scripts/                   # Entry-point scripts (not importable)
│   ├── pretrain.py            # Phase 1: unsupervised pre-training
│   ├── finetune.py            # Phase 2: supervised fine-tuning
│   ├── evaluate.py            # Phase 3: test-set evaluation
│   ├── linear_probe.py        # Run linear probes on frozen features
│   └── run_all.py             # End-to-end: pretrain → finetune → evaluate
├── results_and_figs/          # All outputs go here
│   ├── figures/               # Publication-ready plots
│   ├── tables/                # CSV summary tables
│   └── checkpoints/           # Saved model weights (gitignored)
├── notebooks/                 # Exploratory Jupyter notebooks (optional)
└── vendor/
    └── lpl_zenke/             # Upstream LPL code, unmodified (see VERSION.txt)
```

## What "temporal" means in this project

Upstream LPL on CIFAR-10 has **no temporal stream**. The two "consecutive time steps" that the predictive (pull) loss compares are two augmented views of the same image (`vendor/lpl_zenke/models/module.py`, `LPL.loss_step`). There is no temporal window parameter. Upstream only has real image sequences for Shapes3D (`datasets/sequence_generator.py`).

- **Scramble ablation** = upstream's `--shuffle_positives`: each image's first view is paired with a different image's second view. The Hebbian (variance) and decorrelation terms are unchanged.
- LPL and SimCLR receive the **same kind of view pairs** on CIFAR-10. They differ in the loss (predictive + Hebbian + decorrelation vs contrastive) and in layer-local vs end-to-end training, not in what input structure they see.

## Conventions

### Python

- **Python 3.10+**
- **Formatter/linter:** Ruff (`ruff check .` and `ruff format .`). `vendor/` is excluded in `pyproject.toml`.
- **Imports:** sorted by ruff (isort-compatible). stdlib → third-party → local.
- **Type hints:** use them on function signatures. No need for full mypy compliance.
- **Docstrings:** Google style.

### Naming

- **Files:** `snake_case.py`
- **Classes:** `PascalCase` (e.g., `OjaRule`, `LPLPretrainer`)
- **Functions/variables:** `snake_case`
- **Constants:** `UPPER_SNAKE_CASE`
- **Configs:** `snake_case.yaml`
- **Figures:** `{experiment}_{metric}_{date}.pdf` (e.g., `data_efficiency_accuracy_20261015.pdf`). PDF via `savefig(..., format='pdf')`; PNG only for quick inspection.

### Configs (YAML)

Configs are plain YAML files loaded with `src/utils/config.py`. No Hydra. Shared defaults live in `base.yaml`; each condition YAML overrides them at load time. The condition config is the single source of truth for which method runs (`pretrain.method`).

Sections: `pretrain`, `finetune`, `model`, `data`, `seed`. Labeled set sizes go in `finetune.labeled_counts` and span 10 to 50,000.

**Hyperparameters (epochs, learning rates, batch sizes, number of seeds, the labeled-count grid) are chosen from pilot runs and recorded in the condition YAML. This file does not fix them.**

### Paths

Data and output roots come from environment variables when set (the cluster sets them; see CLUSTER.md). Otherwise they default to repo-relative paths via `pathlib.Path`. Don't hardcode absolute paths.

### Experiment Tracking (MLflow)

- All training scripts log to MLflow. Tracking URI comes from `MLFLOW_TRACKING_URI`; if unset, local `./mlruns`.
- **Experiment names:** `pretrain/{method}`, `finetune/{method}_n{labeled_count}`, `ablation/lpl_shuffled`
- Log: loss curves, final accuracy, config as params, and the checkpoint **path** (checkpoints are stored once, in `results_and_figs/checkpoints/`, not copied as MLflow artifacts).

### Git

- **Branch:** work on `main` for now (solo project).
- **Commits:** short imperative messages. Prefix with scope when helpful:
  - `data: add two-view loader`
  - `exp: run oja pretrain sweep`
  - `analysis: add CKA comparison`
  - `fig: update data efficiency curves`
- Ignored paths are in `.gitignore` (checkpoints, data, mlruns, caches, venvs, CLUSTER.md).

### Scripts

All scripts are run from the repo root:

```bash
python scripts/pretrain.py --config configs/lpl.yaml
python scripts/finetune.py --config configs/lpl.yaml --labeled_n 100 --seed 1
python scripts/evaluate.py --checkpoint results_and_figs/checkpoints/lpl_n100_s1.pt
```

Scripts import from `src/`, never from each other.

### Vendored Code

Upstream LPL (`fmi-basel/latent-predictive-learning`) lives in `vendor/lpl_zenke/`, unmodified. `VERSION.txt` records the commit. If updating from upstream, copy files in and update `VERSION.txt`.

- **Upstream needs an old stack** (torch 1.11, pytorch-lightning 1.6, lightning-bolts 0.5) that does not install on Python 3.11. To run upstream as-is, use the legacy env `.venv-upstream` (Python 3.10, `vendor/lpl_zenke/requirements-legacy.txt`). Run from inside `vendor/lpl_zenke/`, since upstream uses top-level imports (`from models.encoders import ...`). Upstream writes to `~/data/datasets` and `~/data/lpl`.
- **Our pipeline (main env)** imports only upstream's pure-PyTorch modules: `models/encoders.py` and `metrics/losses.py`. The LPL loss logic in `models/module.py` is tied to Lightning, so `src/learning_rules/lpl.py` re-implements it and must have a parity test against `module.py` (run in the legacy env).

## Key Design Decisions

1. **Four conditions are compared together** (random, oja, lpl, simclr). The ablation adds `lpl_shuffled`.

2. **One shared backbone for all conditions.** It is not yet chosen. Upstream LPL is validated only on layer-local VGG-11 (`VGG11Encoder`), so a different backbone needs its own LPL sanity check.

3. **Labeled subset sampling is seeded and deterministic.** Given `(seed, n)`, the same subset is always selected. Subsets are class-balanced (n/10 per class) and nested across n for a given seed. This lives in `src/data/cifar.py`.

4. **Pre-training and fine-tuning are separate scripts.** Pre-trained weights are saved to `results_and_figs/checkpoints/` and loaded by fine-tuning scripts, so pre-training isn't re-run for each data level.

## Experimental Conditions Quick Reference

| Condition | Config | Pre-training | Bio-plausible? |
|-----------|--------|-------------|----------------|
| Random    | `random_init.yaml` | None (Kaiming init) | N/A |
| Oja       | `oja.yaml` | Oja's rule on unlabeled CIFAR-10 | Yes |
| LPL       | `lpl.yaml` | LPL on augmented view pairs | Yes |
| SimCLR    | `simclr.yaml` | Contrastive learning | No |
| LPL-shuffled (ablation) | `lpl_shuffled.yaml` | LPL with `shuffle_positives` | Yes |

## What Not To Do

- Don't hardcode paths (see Paths).
- Don't put experiment results in the codebase (except `results_and_figs/`). Heavy files go in checkpoints (gitignored).
- Don't modify vendored code in `vendor/`. Wrap or re-implement it in `src/learning_rules/`.
- Don't import between scripts. Shared logic goes in `src/`.
