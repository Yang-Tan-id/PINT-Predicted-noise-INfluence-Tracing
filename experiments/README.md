# Numbered experiment workflow

Every experiment folder exposes the same ordered entrypoints:

```text
00_prepare_dataset.py
01_train_model.py
```

The filename determines the execution order and the parent folder determines
the experiment. For example:

```bash
export PINT_DATA_ROOT=/path/to/pint-data
export PINT_OUTPUT_ROOT=/path/to/pint-outputs
python -m pip install -e ".[data]"

cd experiments/shape3d
python 00_prepare_dataset.py
python 01_train_model.py --dry-run
```

Dataset preparation is connected and keeps its terminal progress bar. The
training entrypoints currently resolve and display the fixed dataset path,
output path, and checkpoint count. A non-dry training run stops with an
explicit message until that experiment's original training backend has been
migrated and validated in this repository.

Shared implementations remain under `src/pint/`; the numbered files are thin,
identical entrypoints rather than four copies of the experiment logic.
