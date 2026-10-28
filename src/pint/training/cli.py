from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from pint.config import EXPERIMENT_NAMES, load_experiment, repo_root
from pint.datasets.paths import data_root
from pint.training.model_versions import ARTBENCH_MODEL_ID, ARTBENCH_MODEL_REVISION


def output_root() -> Path:
    value = os.environ.get("PINT_OUTPUT_ROOT")
    if value:
        return Path(value).expanduser().resolve()
    return repo_root() / "outputs"


def training_plan(experiment: str) -> dict[str, object]:
    config = load_experiment(experiment)
    dataset_paths = {
        "x3": data_root() / "x3" / "67_50000.csv",
        "shape3d": data_root() / "shape3d" / "20000" / "dataset.npz",
        "celeba": data_root() / "celeba" / "10000" / "dataset.npz",
        "artbench2": data_root() / "raw" / "artbench2" / "ArtBench-10.csv",
    }
    backends = {
        "x3": "pytorch_pixel_ddpm",
        "shape3d": "jax_pixel_ddpm",
        "celeba": "jax_pixel_ddpm_unconditional",
        "artbench2": "minisd_lora",
    }
    plan = {
        "experiment": experiment,
        "dataset": str(dataset_paths[experiment]),
        "output": str(output_root() / experiment / "model_training"),
        "checkpoint_count": config["training"]["checkpoint_count"],
        "progress_bar": True,
        "backend": backends[experiment],
        "backend_status": "ready",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
    }
    if experiment == "artbench2":
        plan.update(
            pretrained_model=ARTBENCH_MODEL_ID,
            pretrained_model_revision=ARTBENCH_MODEL_REVISION,
        )
    return plan


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train one PINT experiment model")
    parser.add_argument("--experiment", required=True, choices=EXPERIMENT_NAMES)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved training plan without starting training",
    )
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument(
        "--gpu-id",
        type=int,
        help="physical CUDA device ID to expose to the selected training backend",
    )
    gpu_group.add_argument(
        "--gpu-ids",
        help="comma-separated physical CUDA device IDs for a multi-GPU backend",
    )
    args, backend_args = parser.parse_known_args(argv)
    if args.gpu_id is not None:
        if args.gpu_id < 0:
            parser.error("--gpu-id must be a non-negative integer")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    elif args.gpu_ids is not None:
        values = [value.strip() for value in args.gpu_ids.split(",")]
        if not values or any(not value.isdigit() for value in values):
            parser.error("--gpu-ids must be a comma-separated list of non-negative integers")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(values)
    plan = training_plan(args.experiment)
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return 0

    if "--output-dir" not in backend_args:
        backend_args.extend(["--output-dir", str(plan["output"])])
    if args.experiment == "x3":
        from pint.training.x3_training import main as train_x3

        return train_x3(backend_args)
    if args.experiment in ("shape3d", "celeba"):
        from pint.training.jax_training import main as train_jax

        return train_jax(["--experiment", args.experiment, *backend_args])

    from pint.training.backends.artbench_lora import main as train_artbench

    return int(train_artbench(backend_args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
