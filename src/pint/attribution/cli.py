from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from pint.attribution.common import merge_shards
from pint.config import EXPERIMENT_NAMES
from pint.query_gradients.common import timestep_count_for
from pint.training.cli import output_root


def attribution_plan(
    experiment: str, *, batch_size: int | None = None, scope: str | None = None,
    adamw_enabled: bool = True,
) -> dict[str, object]:
    if batch_size is None:
        batch_size = 2 if experiment == "artbench2" else 8
    if scope is None:
        scope = "paper_5k" if experiment in ("shape3d", "celeba") else "all"
    root = output_root() / experiment
    timestep_count = timestep_count_for(experiment)
    return {
        "experiment": experiment,
        "method": "pint",
        "model_dir": str(root / "model_training"),
        "query_manifest": str(root / "queries" / "manifest.json"),
        "query_gradients": str(root / "query_gradients" / f"timesteps_{timestep_count}"),
        "output": str(root / "attribution" / f"timesteps_{timestep_count}"),
        "timesteps": timestep_count,
        "mc_samples": 1,
        "mc_pairing": "shared with query gradients",
        "scope": scope,
        "batch_size": batch_size,
        "datapoint_sharded": True,
        "progress_bar": True,
        "parameter_sources": ["raw_gradient", "adamw_full"] if adamw_enabled else ["raw_gradient"],
        "score_normalization": "raw" if experiment == "x3" else "query_train_l2",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute PINT datapoint attribution")
    parser.add_argument("--experiment", required=True, choices=EXPERIMENT_NAMES)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--scope")
    parser.add_argument("--no-adamw", action="store_true", help="compute raw-gradient attribution only")
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--gpu-id", type=int)
    gpu_group.add_argument("--gpu-ids")
    parser.add_argument("--dry-run", action="store_true")
    args, backend_args = parser.parse_known_args(argv)
    if any(value == "--timesteps" or value.startswith("--timesteps=") for value in backend_args):
        parser.error("attribution timesteps are fixed by the query-gradient manifest")
    if any(value == "--mc-samples" or value.startswith("--mc-samples=") for value in backend_args):
        parser.error("attribution MC samples are fixed to the query-gradient run")
    plan = attribution_plan(
        args.experiment, batch_size=args.batch_size, scope=args.scope,
        adamw_enabled=not args.no_adamw,
    )
    if int(plan["batch_size"]) <= 0:
        parser.error("--batch-size must be positive")
    gpu_ids: list[int] = []
    if args.gpu_id is not None:
        if args.gpu_id < 0:
            parser.error("--gpu-id must be non-negative")
        gpu_ids = [args.gpu_id]
    elif args.gpu_ids is not None:
        fields = [field.strip() for field in args.gpu_ids.split(",")]
        if not fields or any(not field.isdigit() for field in fields):
            parser.error("--gpu-ids must contain comma-separated non-negative integers")
        gpu_ids = [int(field) for field in fields]
        if len(gpu_ids) != len(set(gpu_ids)):
            parser.error("--gpu-ids cannot contain duplicates")
    shard_count = max(1, len(gpu_ids))
    plan["gpu_ids"] = gpu_ids
    plan["datapoint_shards"] = shard_count
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return 0

    modules = {
        "x3": "pint.attribution.x3",
        "shape3d": "pint.attribution.jax_pixel",
        "celeba": "pint.attribution.jax_pixel",
        "artbench2": "pint.attribution.artbench2",
    }
    shared = [
        "--experiment", args.experiment,
        "--model-dir", str(plan["model_dir"]),
        "--query-manifest", str(plan["query_manifest"]),
        "--query-gradient-dir", str(plan["query_gradients"]),
        "--output-dir", str(plan["output"]),
        "--scope", str(plan["scope"]),
        "--batch-size", str(plan["batch_size"]),
        *(["--no-adamw"] if args.no_adamw else []),
        *backend_args,
    ]
    workers = []
    if shard_count > 1:
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
    else:
        if gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        module = __import__(modules[args.experiment], fromlist=["main"])
        module.main([*shared, "--shard-index", "0", "--shard-count", "1"])
    output = merge_shards(Path(str(plan["output"])), shard_count)
    print(f"merged attribution scores: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
