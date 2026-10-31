from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

from pint.baselines.common import BASELINE_METHODS
from pint.config import EXPERIMENT_NAMES
from pint.evaluation.common import (
    LDS_MODEL_COUNT, LDS_SUBSET_SIZES, merge_true_f, prepare_artbench_jobs,
    prepare_lds_jobs,
)
from pint.training.cli import output_root


def _gpu_ids(parser: argparse.ArgumentParser, gpu_id: int | None, text: str | None) -> list[int | None]:
    if gpu_id is not None:
        if gpu_id < 0:
            parser.error("--gpu-id must be non-negative")
        return [gpu_id]
    if text is None:
        return [None]
    fields = [field.strip() for field in text.split(",")]
    if not fields or any(not field.isdigit() for field in fields):
        parser.error("--gpu-ids must be comma-separated non-negative integers")
    values = [int(field) for field in fields]
    if len(values) != len(set(values)):
        parser.error("--gpu-ids cannot contain duplicates")
    return values


def _chunks(items, count):
    quotient, remainder = divmod(len(items), count)
    start = 0
    out = []
    for index in range(count):
        size = quotient + (index < remainder)
        out.append(items[start : start + size])
        start += size
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train evaluation models and compute true-f targets")
    parser.add_argument("--experiment", required=True, choices=EXPERIMENT_NAMES)
    parser.add_argument("--method", choices=("pint", *BASELINE_METHODS), default="pint")
    parser.add_argument("--projection-dim", type=int, default=4096)
    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument("--metric-batch-size", type=int, default=16)
    parser.add_argument(
        "--parameter-source", choices=("raw", "adamw", "both"), default="both",
        help="ArtBench-2 only: attribution score set(s) used for removal",
    )
    gpu = parser.add_mutually_exclusive_group()
    gpu.add_argument("--gpu-id", type=int)
    gpu.add_argument("--gpu-ids")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args, backend_args = parser.parse_known_args(argv)
    if args.projection_dim <= 0 or args.topk <= 0 or args.metric_batch_size <= 0:
        parser.error("--projection-dim, --topk, and --metric-batch-size must be positive")
    gpus = _gpu_ids(parser, args.gpu_id, args.gpu_ids)
    root = output_root() / args.experiment / "evaluation"
    if args.experiment == "artbench2":
        sources = ("raw", "adamw") if args.parameter_source == "both" else (args.parameter_source,)
        jobs = prepare_artbench_jobs(
            root, method=args.method, projection_dim=args.projection_dim,
            parameter_sources=sources, topk=args.topk,
        )
        plan = {
            "kind": "topk_removal", "method": args.method, "topk": args.topk,
            "parameter_sources": list(sources), "models": len(jobs),
        }
    else:
        if args.method != "pint":
            parser.error("--method is used only by ArtBench-2 top-k removal evaluation")
        if args.parameter_source != "both":
            parser.error("--parameter-source is used only by ArtBench-2")
        jobs = prepare_lds_jobs(args.experiment, root)
        plan = {
            "kind": "lds", "models": LDS_MODEL_COUNT,
            "subset_size": LDS_SUBSET_SIZES[args.experiment],
            "subset_group_seeds": [0, 1, 2], "subsets_per_seed": 64,
        }
    plan.update(
        experiment=args.experiment, gpu_ids=[gpu for gpu in gpus if gpu is not None],
        output=str(root), progress_bar=True, eta=True,
        metrics=(
            ["endpoint_mse", "trajectory_mse", "pixel_l2", "clip_distance", "dino_distance", "lpips"]
            if args.experiment == "artbench2"
            else ["trajectory_deviation"]
        ),
        parameter_source="ema" if args.experiment != "artbench2" else "lora",
    )
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return 0

    specs = root / "job_specs"
    logs = root / "logs"
    specs.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    pending = []
    for job in jobs:
        metrics = job.output_dir / "true_f.npz"
        assets = job.output_dir / "evaluation_assets"
        artbench_complete = all(
            (assets / name).is_file()
            for name in ("reference.png", "candidate.png", "trajectories.npz")
        )
        if metrics.is_file() and not args.force and (
            args.experiment != "artbench2" or artbench_complete
        ):
            continue
        payload = {**job.as_json(), "backend_args": backend_args, "force": args.force}
        path = specs / f"{job.job_id}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        pending.append((job, path))

    from tqdm.auto import tqdm

    progress = tqdm(
        total=len(jobs), initial=len(jobs) - len(pending),
        desc=f"{args.experiment} evaluation models", unit="model", dynamic_ncols=True,
    )
    lock = threading.Lock()
    stopped = threading.Event()

    def run_worker(gpu_id, assigned):
        environment = os.environ.copy()
        if gpu_id is not None:
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        for job, spec_path in assigned:
            if stopped.is_set():
                return
            log_path = logs / f"{job.job_id}.log"
            with log_path.open("a", encoding="utf-8") as handle:
                result = subprocess.run(
                    [sys.executable, "-m", "pint.evaluation.worker", "--job-spec", str(spec_path)],
                    env=environment, stdout=handle, stderr=subprocess.STDOUT,
                )
            if result.returncode:
                stopped.set()
                raise subprocess.CalledProcessError(result.returncode, result.args)
            with lock:
                progress.update(1)
                progress.set_postfix(last=job.job_id)

    try:
        assignments = _chunks(pending, len(gpus))
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(run_worker, gpu_id, assigned) for gpu_id, assigned in zip(gpus, assignments)]
            for future in futures:
                future.result()
    finally:
        progress.close()
    if args.experiment == "artbench2":
        if gpus[0] is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[0])
        from pint.evaluation.artbench_perceptual import add_perceptual_metrics

        add_perceptual_metrics(jobs, batch_size=args.metric_batch_size)
    merged_root = root / ("topk_removal" if args.experiment == "artbench2" else "lds")
    merged = merge_true_f(
        jobs,
        merged_root / args.method / "true_f.npz"
        if args.experiment == "artbench2"
        else merged_root / "true_f.npz",
    )
    print(f"evaluation complete: {merged}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
