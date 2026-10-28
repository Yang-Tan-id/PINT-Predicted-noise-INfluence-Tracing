# Dataset download and preparation

Put downloaded source files under the repository's `data/raw/` directory. The
whole `data/` directory is ignored by Git.

```bash
mkdir -p data/raw/shape3d data/raw/celeba data/raw/artbench2
```

## x3

x3 is generated locally and has no external download. Build the fixed 50,000
row dataset (seed 67) with:

```bash
python experiments/x3/00_prepare_dataset.py
```

The CSV and metadata are written under `data/x3/`. The progress bar
tracks generated samples.

## Shape3D

The official Google DeepMind repository describes the dataset and links to the
Google Cloud Storage bucket containing `3dshapes.h5`:

- Dataset documentation: <https://github.com/google-deepmind/3d-shapes>
- Official storage bucket: <https://console.cloud.google.com/storage/browser/3d-shapes>

Download `3dshapes.h5` and place it at:

```text
data/raw/shape3d/3dshapes.h5
```

Then build the fixed balanced 20,000-image experiment dataset:

```bash
python -m pip install -e ".[data]"

python experiments/shape3d/00_prepare_dataset.py
```

The preparation procedure uses seed 42 and selects five examples from each of
4,000 factor groups. It writes `dataset.npz`, the original source indices, and
preparation metadata.

## CelebA

CelebA is available from the official Chinese University of Hong Kong dataset
page:

- Official download page and agreement: <https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>

CelebA is released for non-commercial research use. Review and comply with the
agreement on the official page. Do not commit, publish, or redistribute CelebA
images through this repository.

Download these files from the official page:

```text
img_align_celeba.zip
list_eval_partition.txt
```

The following file is optional and is needed only for the attribute-conditioned
dataset variant:

```text
list_attr_celeba.txt
```

Place the downloaded files under:

```text
data/raw/celeba/
```

Build the fixed unconditional 10,000-image experiment dataset with:

```bash
python -m pip install -e ".[data]"

python experiments/celeba/00_prepare_dataset.py
```

The preparation procedure uses the official training partition, seed 42,
center-crops each selected image to 140x140, and resizes it to 64x64 with
bicubic interpolation.

## ArtBench-2

Place the ArtBench metadata and image-folder export at these standard paths:

```text
data/raw/artbench2/ArtBench-10.csv
data/raw/artbench2/artbench-10-imagefolder/<label>/<filename>
```

The CSV must include `name`, `label`, and `split` columns. Prepare the fixed
5,000 training and 1,000 validation rows with:

```bash
python experiments/artbench2/00_prepare_dataset.py
```

This writes indices and readable manifests under
`data/artbench2/indices/`. It validates selected image paths with a
progress bar. Use `--skip-image-check` only when preparing indices before the
image files have been placed.

## Attribution candidate manifests

Dataset preparation and attribution candidate selection are separate. The
paper experiments use the fixed manifests checked into this repository:

```text
dataset_manifests/shape3d/attribution_5k_indices.npy
dataset_manifests/celeba/attribution_5k_indices.npy
dataset_manifests/celeba/attribution_5k_mask.npy
```

Verify or deterministically recreate them with:

```bash
python scripts/prepare_attribution_scope.py --experiment shape3d
python scripts/prepare_attribution_scope.py --experiment celeba
python -m pint.cli validate --check-data
```

These small files contain only integer indices or a Boolean mask. They do not
contain Shape3D or CelebA image pixels. Use attribution scope `all` instead of
`paper_5k` to compute attribution over every prepared training datapoint.
