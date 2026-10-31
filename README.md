# PINT: Predicted-noise INfluence Tracing

This repository contains the reproducible experiment framework for four
diffusion data-attribution experiments:

- `x3`: 100 prompted queries, 50 checkpoints, 192 LDS subsets, and top-k removal
  evaluation.
- `shape3d`: 100 prompted queries, 50 checkpoints, and 192 LDS subsets.
- `celeba`: 100 prompted queries, 50 checkpoints, and 192 LDS subsets.
- `artbench2`: 60 prompted queries, 25 checkpoints, and top-k removal evaluation.

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
backends.

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

- Pixel experiments save 50 evenly spaced checkpoints by default; ArtBench-2
  saves 25. The final epoch is always included. Change the count with
  `--num-checkpoints N`.
- GPU selection: `--gpu-id N`; JAX multi-GPU: `--gpu-ids 0,1,2,3`.

Checkpoints are written under `outputs/<dataset>/model_training/`.

### 8. Sample queries

```bash
python 02_sample_queries.py --gpu-id 0
```

By default, this samples 100 queries for x3, Shape3D, and CelebA, or 60 for
ArtBench-2. Results are saved to `outputs/<dataset>/queries/`.

- Change the number of queries with `--num-queries N`.
- Change the starting sampling seed with `--seed N`. The default is the paper seed.

### 9. Compute query gradients

```bash
python 03_compute_query_gradients.py --gpu-id 0
```

This runs PINT over the complete query bank in checkpoint-major order. Adjust
query batching with `--batch-size N` and change the default 4096-dimensional
projection with `--projection-dim N`. PINT uses 100 timesteps for x3 and
Shape3D, and 50 for CelebA and ArtBench-2, matching query sampling. Use
`--gpu-ids 0,1,2,3` to split consecutive checkpoint ranges across GPUs.

For baseline queries, set the baseline tag to `gas`, `tracincp`, or `retrac`:

```bash
python ../../scripts/methods/01_compute_queries.py --experiment shape3d --method gas --gpu-id 0
```

### 10. Compute attribution

```bash
python 04_compute_attribution.py --gpu-id 0
```

Use `--batch-size N` to adjust datapoint batching and `--gpu-ids 0,1,2,3` for
multiple GPUs. Raw-gradient and AdamW scores are saved by default; add
`--no-adamw` to save raw-gradient scores only.

Use the same baseline tag for attribution as for its query run:

```bash
python ../../scripts/methods/02_compute_attribution.py --experiment shape3d --method gas --gpu-id 0
```

## Validation

Validate the setup and run the unit tests:

```bash
pint validate
python -m unittest discover -s tests -v
```

## Notes

- ArtBench-2 uses
  [`lambdalabs/miniSD-diffusers`](https://huggingface.co/lambda/miniSD-diffusers)
  at revision `26ed8a9bfbf76f46a6cf60517dde321f900c44ce`. The first run downloads it;
  later runs reuse the Hugging Face cache.
- The `data/` and `outputs/` directories are ignored by Git.
