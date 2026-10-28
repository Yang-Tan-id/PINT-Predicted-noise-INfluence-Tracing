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

## Run an experiment

```bash
python -m pip install -e ".[data]"
cd experiments/shape3d
python 00_prepare_dataset.py
python 01_train_model.py --dry-run
```

Run the numbered files from top to bottom. By default, raw and prepared data go
under `data/`, and experiment outputs go under `outputs/`. Both directories are
ignored by Git. Preparation loops retain their terminal progress bars.

See [`experiments/README.md`](experiments/README.md) for the current workflow
and model-training backend status. See
[`dataset_manifests/README.md`](dataset_manifests/README.md) for the few raw
files that must be downloaded for Shape3D, CelebA, and ArtBench-2.

## Validation

```bash
pint validate
python -m unittest discover -s tests -v
```

Use attribution scope `all` to target every training datapoint. The paper
configurations for Shape3D and CelebA use the fixed `paper_5k` candidate
manifests.

Dataset preparation and attribution candidate selection are separate. Verify or
recreate the fixed paper candidate manifests with:

```bash
python scripts/prepare_attribution_scope.py --experiment shape3d
python scripts/prepare_attribution_scope.py --experiment celeba
```

The current commit establishes configuration, validation, dataset-manifest,
scope-resolution, and launcher interfaces. Experiment backends are migrated
into these interfaces one experiment at a time.
