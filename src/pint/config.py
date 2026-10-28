from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


EXPERIMENT_NAMES = ("x3", "shape3d", "celeba", "artbench2")


def repo_root() -> Path:
    override = os.environ.get("PINT_REPO_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return payload


def experiment_path(name: str, root: Path | None = None) -> Path:
    if name not in EXPERIMENT_NAMES:
        raise ValueError(
            f"Unknown experiment {name!r}; expected one of {', '.join(EXPERIMENT_NAMES)}"
        )
    base = root.resolve() if root is not None else repo_root()
    return base / "experiments" / name / "experiment.yaml"


def load_experiment(name: str, root: Path | None = None) -> dict[str, Any]:
    config = load_yaml(experiment_path(name, root))
    validate_experiment(config, expected_name=name)
    return config


def validate_experiment(config: dict[str, Any], expected_name: str | None = None) -> None:
    required = ("schema_version", "name", "dataset", "queries", "sampling", "training")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Experiment config is missing: {', '.join(missing)}")
    if config["schema_version"] != 1:
        raise ValueError("Only experiment schema_version=1 is supported")
    if expected_name is not None and config["name"] != expected_name:
        raise ValueError(
            f"Experiment name mismatch: expected {expected_name!r}, got {config['name']!r}"
        )
    query_count = config["queries"].get("count")
    if not isinstance(query_count, int) or query_count <= 0:
        raise ValueError("queries.count must be a positive integer")
    checkpoint_count = config["training"].get("checkpoint_count")
    if not isinstance(checkpoint_count, int) or checkpoint_count <= 0:
        raise ValueError("training.checkpoint_count must be a positive integer")
    lds = config.get("lds", {})
    if lds.get("enabled"):
        num_subsets = lds.get("num_subsets")
        if not isinstance(num_subsets, int) or num_subsets <= 0:
            raise ValueError("lds.num_subsets must be positive when LDS is enabled")


def scope_path(
    experiment: str,
    scope: str,
    root: Path | None = None,
) -> Path:
    base = root.resolve() if root is not None else repo_root()
    return base / "experiments" / experiment / "attribution_scopes" / f"{scope}.yaml"
