# Dataset download and preparation

Raw datasets and prepared image arrays are intentionally excluded from Git.
Only experiment definitions, fixed attribution candidate manifests, checksums,
and preparation code are versioned in this repository.

Set one external data root before following the instructions below:

```bash
export PINT_DATA_ROOT=/path/to/pint-data
mkdir -p "$PINT_DATA_ROOT/raw/shape3d" "$PINT_DATA_ROOT/raw/celeba"
```

## Shape3D

The official Google DeepMind repository describes the dataset and links to the
Google Cloud Storage bucket containing `3dshapes.h5`:

- Dataset documentation: <https://github.com/google-deepmind/3d-shapes>
- Official storage bucket: <https://console.cloud.google.com/storage/browser/3d-shapes>

Download `3dshapes.h5` and place it at:

```text
$PINT_DATA_ROOT/raw/shape3d/3dshapes.h5
```

Then build the fixed balanced 20,000-image experiment dataset:

```bash
python -m pip install -e ".[data]"

python scripts/prepare_dataset.py --experiment shape3d \
  --input "$PINT_DATA_ROOT/raw/shape3d/3dshapes.h5" \
  --out-root "$PINT_DATA_ROOT/shape3d/20000"
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
$PINT_DATA_ROOT/raw/celeba/
```

Build the fixed unconditional 10,000-image experiment dataset with:

```bash
python -m pip install -e ".[data]"

python scripts/prepare_dataset.py --experiment celeba \
  --images-zip "$PINT_DATA_ROOT/raw/celeba/img_align_celeba.zip" \
  --partition "$PINT_DATA_ROOT/raw/celeba/list_eval_partition.txt" \
  --out-root "$PINT_DATA_ROOT/celeba/10000"
```

The preparation procedure uses the official training partition, seed 42,
center-crops each selected image to 140x140, and resizes it to 64x64 with
bicubic interpolation.

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
