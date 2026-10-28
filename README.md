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

## Setup and run an experiment

The paper training setup targets Linux, Python 3.11, and NVIDIA GPUs. The JAX
backend is configured for CUDA 12.

### 1. Clone the repository

```bash
git clone https://github.com/Yang-Tan-id/PINT-Predicted-noise-INfluence-Tracing.git
cd PINT-Predicted-noise-INfluence-Tracing
```

Complete setup and dataset download from the repository root. In step 5, enter
the experiment directory you want to run.

### 2. Create a Python environment

```bash
conda create -n pint python=3.11 -y
conda activate pint
python -m pip install --upgrade pip
```

### 3. Install the dependencies

```bash
python -m pip install -r requirements.txt
```

This installs the data-preparation dependencies and all four training
backends. Dataset-specific installation options are documented in
[`experiments/README.md`](experiments/README.md).

### 4. Download the dataset

`x3` is generated locally and requires no download. For Shape3D, CelebA, and
ArtBench-2, download the source files and place them at the documented paths in
[`dataset_manifests/README.md`](dataset_manifests/README.md).

### 5. Prepare the dataset

Choose one experiment and enter its directory. For example:

```bash
cd experiments/shape3d
python 00_prepare_dataset.py
```

Replace `shape3d` with `x3`, `celeba`, or `artbench2` as needed. Step `00`
generates or preprocesses the data and writes it under `data/`.

### 6. Check the training configuration

```bash
python 01_train_model.py --dry-run --gpu-id 0
```

The dry run prints the dataset path, output path, backend, checkpoint count,
and selected CUDA device without loading the model.

### 7. Train the model

```bash
python 01_train_model.py --gpu-id 0
```

Use `--gpu-id N` to select one physical GPU. JAX multi-GPU training accepts,
for example, `--gpu-ids 0,1,2,3`. Checkpoints are written under
`outputs/<dataset>/model_training/`.

ArtBench-2 fine-tunes the Hugging Face model
[`lambdalabs/miniSD-diffusers`](https://huggingface.co/lambda/miniSD-diffusers)
at the pinned revision
`26ed8a9bfbf76f46a6cf60517dde321f900c44ce`. The first ArtBench-2 training run
downloads that exact snapshot automatically and later runs reuse the Hugging
Face cache. Pass `--model` and `--model-revision` only when intentionally
running a different base-model version.

The `data/` and `outputs/` directories are ignored by Git.

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

The numbered training entrypoint dispatches to the paper backend for each
dataset.
