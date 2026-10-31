from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from pint.config import EXPERIMENT_NAMES
from pint.datasets.paths import data_root
from pint.sampling.queries import default_query_count, default_query_seed
from pint.training.cli import output_root


def sampling_plan(
    experiment: str, query_count: int | None = None, seed: int | None = None
) -> dict[str, object]:
    count = default_query_count(experiment) if query_count is None else query_count
    return {
        "experiment": experiment,
        "query_count": count,
        "seed": default_query_seed(experiment) if seed is None else seed,
        "model_dir": str(output_root() / experiment / "model_training"),
        "output": str(output_root() / experiment / "queries"),
        "dataset": str(
            {
                "x3": data_root() / "x3" / "67_50000.csv",
                "shape3d": data_root() / "shape3d" / "20000" / "dataset.npz",
                "celeba": data_root() / "celeba" / "10000" / "dataset.npz",
                "artbench2": data_root() / "artbench2" / "5000",
            }[experiment]
        ),
        "sampler": "ddim_eta0",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sample deterministic PINT queries")
    parser.add_argument("--experiment", required=True, choices=EXPERIMENT_NAMES)
    parser.add_argument("--num-queries", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpu-id", type=int)
    args, backend_args = parser.parse_known_args(argv)
    if args.num_queries is not None and args.num_queries <= 0:
        parser.error("--num-queries must be a positive integer")
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be a non-negative integer")
    if args.gpu_id is not None:
        if args.gpu_id < 0:
            parser.error("--gpu-id must be a non-negative integer")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    plan = sampling_plan(args.experiment, args.num_queries, args.seed)
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return 0

    if "--output-dir" not in backend_args:
        backend_args.extend(["--output-dir", str(plan["output"])])
    if "--model-dir" not in backend_args:
        backend_args.extend(["--model-dir", str(plan["model_dir"])])
    backend_args.extend(["--num-queries", str(plan["query_count"])])
    backend_args.extend(["--seed", str(plan["seed"])])

    if args.experiment == "x3":
        from pint.sampling.x3 import main as sample_x3

        return sample_x3(backend_args)
    if args.experiment in ("shape3d", "celeba"):
        from pint.sampling.jax_pixel import main as sample_jax

        return sample_jax(["--experiment", args.experiment, *backend_args])

    from pint.sampling.artbench2 import main as sample_artbench2

    return sample_artbench2(backend_args)


if __name__ == "__main__":
    raise SystemExit(main())
