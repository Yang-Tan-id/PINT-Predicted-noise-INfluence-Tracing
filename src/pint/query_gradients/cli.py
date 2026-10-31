from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from pint.baselines.common import BASELINE_METHODS
from pint.config import EXPERIMENT_NAMES
from pint.query_gradients.common import normalization_for, timestep_count_for
from pint.training.cli import output_root


def query_gradient_plan(
    experiment: str,
    *,
    batch_size: int = 8,
    projection_dim: int = 4096,
) -> dict[str, object]:
    timesteps = timestep_count_for(experiment)
    root = output_root() / experiment
    return {
        "experiment": experiment,
        "method": "pint",
        "query_manifest": str(root / "queries" / "manifest.json"),
        "model_dir": str(root / "model_training"),
        "output": str(root / "query_gradients" / f"timesteps_{timesteps}"),
        "query_batch_size": batch_size,
        "timesteps": timesteps,
        "projection_dim": projection_dim,
        "normalization": normalization_for(experiment),
        "checkpoint_major": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    method_parser = argparse.ArgumentParser(add_help=False)
    method_parser.add_argument(
        "--method", choices=("pint", *BASELINE_METHODS), default="pint"
    )
    selected_method, _ = method_parser.parse_known_args(argv)
    if selected_method.method != "pint":
        from pint.baselines.query_cli import main as baseline_main

        return baseline_main(argv)

    parser = argparse.ArgumentParser(
        description="Compute PINT gradients for the complete query bank"
    )
    parser.add_argument("--experiment", required=True, choices=EXPERIMENT_NAMES)
    parser.add_argument(
        "--method", choices=("pint", *BASELINE_METHODS), default="pint"
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--projection-dim", type=int, default=4096)
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--gpu-id", type=int)
    gpu_group.add_argument(
        "--gpu-ids",
        help="comma-separated physical GPU IDs; checkpoint pairs are split contiguously",
    )
    parser.add_argument("--dry-run", action="store_true")
    args, backend_args = parser.parse_known_args(argv)
    if any(
        value == "--timesteps" or value.startswith("--timesteps=")
        for value in backend_args
    ):
        parser.error(
            "--timesteps is fixed to match sampling: 100 for x3/Shape3D and "
            "50 for CelebA/ArtBench-2"
        )
    if args.batch_size is None:
        args.batch_size = 2 if args.experiment == "artbench2" else 8
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.projection_dim <= 0:
        parser.error("--projection-dim must be positive")
    gpu_ids: list[int] = []
    if args.gpu_id is not None:
        if args.gpu_id < 0:
            parser.error("--gpu-id must be non-negative")
        gpu_ids = [args.gpu_id]
    elif args.gpu_ids is not None:
        values = [value.strip() for value in args.gpu_ids.split(",")]
        if not values or any(not value.isdigit() for value in values):
            parser.error("--gpu-ids must be a comma-separated list of non-negative integers")
        gpu_ids = [int(value) for value in values]
        if len(gpu_ids) != len(set(gpu_ids)):
            parser.error("--gpu-ids cannot contain duplicate device IDs")

    plan = query_gradient_plan(
        args.experiment,
        batch_size=args.batch_size,
        projection_dim=args.projection_dim,
    )
    if gpu_ids:
        plan["gpu_ids"] = gpu_ids
        plan["checkpoint_shards"] = len(gpu_ids)
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return 0

    shared = [
        "--experiment", args.experiment,
        "--model-dir", str(plan["model_dir"]),
        "--query-manifest", str(plan["query_manifest"]),
        "--output-dir", str(plan["output"]),
        "--batch-size", str(args.batch_size),
        "--projection-dim", str(args.projection_dim),
        *backend_args,
    ]
    if len(gpu_ids) > 1:
        modules = {
            "x3": "pint.query_gradients.x3",
            "shape3d": "pint.query_gradients.jax_pixel",
            "celeba": "pint.query_gradients.jax_pixel",
            "artbench2": "pint.query_gradients.artbench2",
        }
        workers = []
        for shard_index, gpu_id in enumerate(gpu_ids):
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            command = [
                sys.executable,
                "-m",
                modules[args.experiment],
                *shared,
                "--shard-index", str(shard_index),
                "--shard-count", str(len(gpu_ids)),
            ]
            workers.append(subprocess.Popen(command, env=environment))
        for worker in workers:
            return_code = worker.wait()
            if return_code:
                for other in workers:
                    if other.poll() is None:
                        other.terminate()
                for other in workers:
                    if other.poll() is None:
                        other.wait()
                raise subprocess.CalledProcessError(return_code, worker.args)
        return 0

    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
    if args.experiment == "x3":
        from pint.query_gradients.x3 import main as backend_main
    elif args.experiment in ("shape3d", "celeba"):
        from pint.query_gradients.jax_pixel import main as backend_main
    else:
        from pint.query_gradients.artbench2 import main as backend_main
    return int(backend_main(shared) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
