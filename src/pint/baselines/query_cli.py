from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from pint.baselines.common import BASELINE_METHODS, baseline_spec, run_tag
from pint.config import EXPERIMENT_NAMES
from pint.query_gradients.common import default_projection_dim_for
from pint.training.cli import output_root


def query_plan(
    experiment: str, method: str, *, projection_dim: int | None = None,
    batch_size: int | None = None
) -> dict[str, object]:
    if projection_dim is None:
        projection_dim = default_projection_dim_for(experiment)
    spec = baseline_spec(method, experiment)
    tag = run_tag(spec, projection_dim)
    if batch_size is None:
        batch_size = 2 if experiment == "artbench2" else 8
    root = output_root() / experiment
    return {
        "experiment": experiment,
        "method": method,
        "tag": tag,
        "model_dir": str(root / "model_training"),
        "query_manifest": str(root / "queries" / "manifest.json"),
        "output": str(root / "methods" / method / "queries"),
        "timesteps": spec.timestep_count,
        "mc_samples": spec.mc_samples,
        "projection_dim": projection_dim,
        "batch_size": batch_size,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute baseline query features")
    parser.add_argument("--experiment", required=True, choices=EXPERIMENT_NAMES)
    parser.add_argument("--method", required=True, choices=BASELINE_METHODS)
    parser.add_argument("--projection-dim", type=int)
    parser.add_argument("--batch-size", type=int)
    gpu = parser.add_mutually_exclusive_group()
    gpu.add_argument("--gpu-id", type=int)
    gpu.add_argument("--gpu-ids")
    parser.add_argument("--dry-run", action="store_true")
    args, backend_args = parser.parse_known_args(argv)
    if args.projection_dim is None:
        args.projection_dim = default_projection_dim_for(args.experiment)
    if args.projection_dim <= 0:
        parser.error("--projection-dim must be positive")
    plan = query_plan(
        args.experiment,
        args.method,
        projection_dim=args.projection_dim,
        batch_size=args.batch_size,
    )
    if int(plan["batch_size"]) <= 0:
        parser.error("--batch-size must be positive")
    gpu_ids: list[int] = []
    if args.gpu_id is not None:
        gpu_ids = [args.gpu_id]
    elif args.gpu_ids:
        fields = [field.strip() for field in args.gpu_ids.split(",")]
        if any(not field.isdigit() for field in fields):
            parser.error("--gpu-ids must be comma-separated non-negative integers")
        gpu_ids = [int(field) for field in fields]
    if any(value < 0 for value in gpu_ids) or len(gpu_ids) != len(set(gpu_ids)):
        parser.error("GPU IDs must be unique non-negative integers")
    plan["gpu_ids"] = gpu_ids
    plan["checkpoint_shards"] = max(1, len(gpu_ids))
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return 0
    modules = {
        "x3": "pint.baselines.query_x3",
        "shape3d": "pint.baselines.query_jax",
        "celeba": "pint.baselines.query_jax",
        "artbench2": "pint.baselines.query_artbench2",
    }
    shared = [
        "--experiment", args.experiment,
        "--method", args.method,
        "--model-dir", str(plan["model_dir"]),
        "--query-manifest", str(plan["query_manifest"]),
        "--output-dir", str(plan["output"]),
        "--projection-dim", str(args.projection_dim),
        "--batch-size", str(plan["batch_size"]),
        *backend_args,
    ]
    shard_count = max(1, len(gpu_ids))
    if shard_count > 1:
        workers = []
        for shard_index, gpu_id in enumerate(gpu_ids):
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            command = [
                sys.executable, "-m", modules[args.experiment], *shared,
                "--shard-index", str(shard_index), "--shard-count", str(shard_count),
            ]
            workers.append(subprocess.Popen(command, env=environment))
        for worker in workers:
            code = worker.wait()
            if code:
                for other in workers:
                    if other.poll() is None:
                        other.terminate()
                raise subprocess.CalledProcessError(code, worker.args)
        return 0
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
    module = __import__(modules[args.experiment], fromlist=["main"])
    return int(module.main([*shared, "--shard-index", "0", "--shard-count", "1"]) or 0)
