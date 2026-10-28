from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from pint.config import EXPERIMENT_NAMES, load_experiment, repo_root
from pint.datasets.paths import data_root


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
        "artbench2": data_root() / "artbench2" / "indices" / "manifests" / "train.csv",
    }
    return {
        "experiment": experiment,
        "dataset": str(dataset_paths[experiment]),
        "output": str(output_root() / experiment / "model_training"),
        "checkpoint_count": config["training"]["checkpoint_count"],
        "progress_bar": True,
        "backend_status": "pending_migration",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train one PINT experiment model")
    parser.add_argument("--experiment", required=True, choices=EXPERIMENT_NAMES)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved training plan without starting training",
    )
    args = parser.parse_args(argv)
    plan = training_plan(args.experiment)
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return 0
    raise SystemExit(
        "Model training has not been connected to the cleaned repository yet. "
        "Run with --dry-run to inspect the fixed paths/checkpoint count; migrate and validate "
        "the experiment backend before starting a paper run."
    )


if __name__ == "__main__":
    raise SystemExit(main())
