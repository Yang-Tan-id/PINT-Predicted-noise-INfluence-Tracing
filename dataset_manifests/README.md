# Datasets

Install the data dependencies and create the download directories:

```bash
python -m pip install -e ".[data]"
mkdir -p data/raw/{shape3d,celeba,artbench2}
```

Download the external datasets into the following locations. The entire
`data/` directory is ignored by Git.

| Dataset | Download | Required location |
| --- | --- | --- |
| Shape3D | Download `3dshapes.h5` from the [official Google DeepMind page](https://github.com/google-deepmind/3d-shapes#download-data-file-from-google-cloud-storage). | `data/raw/shape3d/3dshapes.h5` |
| CelebA | Accept the terms on the [official CelebA page](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), then download **Align&Cropped Images** (`img_align_celeba.zip`) and **Train/Val/Test Partitions** (`list_eval_partition.txt`). | `data/raw/celeba/img_align_celeba.zip` and `data/raw/celeba/list_eval_partition.txt` |
| ArtBench-2 | From the [official ArtBench repository](https://github.com/liaopeiyuan/artbench#accessing-dataset), download **Metadata** and the 256x256 **ImageFolder** archive. Rename the metadata file to `ArtBench-10.csv` if needed, and extract the image archive. | `data/raw/artbench2/ArtBench-10.csv` and `data/raw/artbench2/artbench-10-imagefolder/<label>/<filename>` |

`x3` requires no download. After placing the external files, run the
`00_prepare_dataset.py` file inside the corresponding experiment directory;
it will prepare the dataset for that experiment. For example:

```bash
python experiments/shape3d/00_prepare_dataset.py
```

For `x3`, running `experiments/x3/00_prepare_dataset.py` generates and prepares
the dataset automatically.

CelebA is licensed for non-commercial research use and must not be
redistributed. The checked-in attribution manifests contain indices only, not
dataset images.
