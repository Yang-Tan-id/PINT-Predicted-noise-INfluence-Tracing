# PINT: Predicted-noise INfluence Tracing

This repository contains the reproducible experiment framework for four
diffusion data-attribution experiments:

- `x3`: 100 queries, 50 checkpoints, and 192 LDS subsets.
- `shape3d`: 100 queries, 50 checkpoints, and 192 LDS subsets.
- `celeba`: 100 queries, 50 checkpoints, and 192 LDS subsets.
- `artbench2`: 60 queries, 25 checkpoints, and top-k removal evaluation.

Experiment definitions and identical numbered entrypoints live under
`experiments/`. Shared implementation remains outside those folders under
`src/`, so experiment logic is not copied four times.

## Repository policy

Datasets, checkpoints, projected gradients, trajectories, and complete outputs
are not committed to Git. Dataset preparation definitions and split manifests
live under `dataset_manifests/`.

Set external storage locations before running an experiment:

```bash
export PINT_DATA_ROOT=/path/to/pint-data
export PINT_OUTPUT_ROOT=/path/to/pint-outputs
```

## Setup and validation

```bash
python -m pip install -e .
pint validate
python -m unittest discover -s tests -v
```

Display a resolved experiment definition:

```bash
pint show-config --experiment shape3d
```

Preview a run without executing a backend:

```bash
python scripts/compute_attribution.py \
  --experiment shape3d \
  --scope paper_5k \
  --dry-run
```

Use `--scope all` to target every training datapoint. The paper configurations
for Shape3D and CelebA use the fixed `paper_5k` candidate manifests.

## Dataset preparation

Each experiment folder has the same ordered filenames. Install the optional
preparation dependencies, then run the entrypoint in the dataset folder:

```bash
python -m pip install -e ".[data]"

python experiments/shape3d/00_prepare_dataset.py
python experiments/shape3d/01_train_model.py --dry-run
```

The preparation scripts use the standard paths documented below, so ordinary
runs need no dataset-specific path arguments. All preparation loops retain
their terminal progress bars. See [`experiments/README.md`](experiments/README.md)
for the numbered workflow and current model-training backend status.

Dataset preparation and attribution candidate selection are separate. Verify or
recreate the fixed paper candidate manifests with:

```bash
python scripts/prepare_attribution_scope.py --experiment shape3d
python scripts/prepare_attribution_scope.py --experiment celeba
```

See [`dataset_manifests/README.md`](dataset_manifests/README.md) for official
download locations, required files, data-use notes, and complete preparation
commands.

The current commit establishes configuration, validation, dataset-manifest,
scope-resolution, and launcher interfaces. Experiment backends are migrated
into these interfaces one experiment at a time.
