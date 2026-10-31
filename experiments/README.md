# Numbered experiment workflow

Every experiment folder exposes the same ordered entrypoints:

```text
00_prepare_dataset.py
01_train_model.py
02_sample_queries.py
03_compute_query_gradients.py
04_compute_attribution.py
```

The filename determines the execution order and the parent folder determines
the experiment. For example:

```bash
python -m pip install -r requirements.txt

cd experiments/shape3d
python 00_prepare_dataset.py
python 01_train_model.py
python 02_sample_queries.py
python 03_compute_query_gradients.py
python 04_compute_attribution.py
```

No path configuration is required. Data defaults to the repository's `data/`
directory and outputs default to `outputs/`; both are ignored by Git.

Training defaults:

- x3, Shape3D, and CelebA: pixel diffusion models with 50 checkpoints.
- ArtBench-2: miniSD LoRA with 25 checkpoints, using
  `lambdalabs/miniSD-diffusers` at revision
  `26ed8a9bfbf76f46a6cf60517dde321f900c44ce`.

Run `python 01_train_model.py --dry-run` to check the configuration before
training. The ArtBench-2 model is downloaded automatically on its first run.
