from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from pint.datasets.paths import data_root
from pint.training.cli import output_root


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate one evaluation model")
    parser.add_argument("--job-spec", type=Path, required=True)
    args = parser.parse_args(argv)
    spec = json.loads(args.job_spec.read_text(encoding="utf-8"))
    experiment = str(spec["experiment"])
    model_dir = Path(spec["output_dir"])
    train_indices = Path(spec["train_indices"])
    metrics_path = model_dir / "true_f.npz"
    if metrics_path.is_file() and not bool(spec.get("force", False)):
        assets = model_dir / "evaluation_assets"
        if experiment != "artbench2" or (
            (assets / "reference.png").is_file()
            and (assets / "candidate.png").is_file()
            and (assets / "trajectories.npz").is_file()
        ):
            return 0
    backend_args = list(spec.get("backend_args", []))
    model_dir.mkdir(parents=True, exist_ok=True)
    if experiment == "x3":
        from pint.evaluation.metrics_x3 import evaluate
        from pint.training.x3_training import main as train

        trained = bool(list((model_dir / "base" / "prompted").glob("epoch_*.pt")))
        if not trained:
            train([
                "--output-dir", str(model_dir), "--family", "prompted",
                "--num-checkpoints", "1", "--train-indices", str(train_indices),
                "--lds-seed", str(spec["subset_seed"]),
                "--subset-id", str(int(spec["subset_id"]) % 64), *backend_args,
            ])
        evaluate(
            model_dir,
            output_root() / "x3" / "model_training",
            output_root() / "x3" / "queries" / "manifest.json",
            metrics_path,
        )
    elif experiment in ("shape3d", "celeba"):
        from pint.evaluation.metrics_jax import evaluate
        from pint.training.jax_training import main as train

        trained = bool(list(model_dir.glob("seed_*_epoch_*.ckpt")))
        if not trained:
            train([
                "--experiment", experiment, "--output-dir", str(model_dir),
                "--num-checkpoints", "1", "--num-devices", "1",
                "--train-indices", str(train_indices), *backend_args,
            ])
        size = "20000" if experiment == "shape3d" else "10000"
        evaluate(
            experiment,
            model_dir,
            output_root() / experiment / "model_training",
            output_root() / experiment / "queries" / "manifest.json",
            data_root() / experiment / size / "dataset.npz",
            metrics_path,
        )
    else:
        from pint.evaluation.metrics_artbench2 import evaluate
        from pint.training.backends.artbench_lora import main as train

        if not (model_dir / "final").is_dir():
            train([
                "--output-dir", str(model_dir), "--index-path", str(train_indices),
                "--num-checkpoints", "1", *backend_args,
            ])
        evaluate(
            model_dir,
            output_root() / "artbench2" / "model_training",
            output_root() / "artbench2" / "queries" / "manifest.json",
            int(spec["query_id"]),
            metrics_path,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
