# Numbered experiment workflow

Every experiment folder exposes the same ordered entrypoints:

```text
00_prepare_dataset.py
01_train_model.py
```

The filename determines the execution order and the parent folder determines
the experiment. For example:

```bash
python -m pip install -r requirements.txt

cd experiments/shape3d
python 00_prepare_dataset.py
python 01_train_model.py
```

No path configuration is required. Data defaults to the repository's `data/`
directory and outputs default to `outputs/`; both are ignored by Git.

Dataset preparation and model training keep terminal progress bars. The root
`requirements.txt` installs all four experiment backends. To install only one
backend instead, use:

```bash
python -m pip install -e ".[data,train-x3]"       # x3
python -m pip install -e ".[data,train-jax]"      # Shape3D or CelebA
python -m pip install -e ".[data,train-artbench]" # ArtBench-2
```

The migrated paper backends are:

- x3: PyTorch pixel DDPM, prompted and unprompted, 50 checkpoints.
- Shape3D: conditional JAX pixel DDPM, 50 checkpoints.
- CelebA: unconditional JAX pixel DDPM, 50 checkpoints.
- ArtBench-2: miniSD LoRA fine-tuning, 25 checkpoints.

ArtBench-2 uses `lambdalabs/miniSD-diffusers` pinned to Hugging Face revision
`26ed8a9bfbf76f46a6cf60517dde321f900c44ce`. It is downloaded automatically
on the first training run and then loaded from the Hugging Face cache.

Use `python 01_train_model.py --dry-run` to print resolved paths without
loading a training framework or starting a run. Additional options are passed
to the selected backend.

Select one physical CUDA device consistently across all four experiments with:

```bash
python experiments/shape3d/01_train_model.py --gpu-id 2
```

The selected physical GPU is exposed to the backend as device 0. For a JAX
multi-GPU run, pass a comma-separated list such as `--gpu-ids 0,1,2,3`. Use
`--device cpu` for an x3 or JAX CPU run.

Shared implementations remain under `src/pint/`; the numbered files are thin,
identical entrypoints rather than four copies of the experiment logic.
