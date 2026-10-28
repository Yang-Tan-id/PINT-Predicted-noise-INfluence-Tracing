from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from pint.config import repo_root


PAPER_SCOPES = {
    "shape3d": {"dataset_size": 20_000, "scope_size": 5_000, "seed": 42, "mask": False},
    "celeba": {"dataset_size": 10_000, "scope_size": 5_000, "seed": 42, "mask": True},
}


def build_scope_indices(dataset_size: int, scope_size: int, seed: int) -> np.ndarray:
    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive")
    if scope_size <= 0 or scope_size > dataset_size:
        raise ValueError("scope_size must be in [1, dataset_size]")
    return np.random.default_rng(seed).choice(
        dataset_size, size=scope_size, replace=False
    ).astype(np.int64)


def build_scope_mask(dataset_size: int, indices: np.ndarray) -> np.ndarray:
    mask = np.zeros(dataset_size, dtype=np.bool_)
    mask[np.asarray(indices, dtype=np.int64)] = True
    return mask


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_paper_scope(experiment: str, output_root: Path, force: bool = False) -> dict:
    definition = PAPER_SCOPES[experiment]
    output_root.mkdir(parents=True, exist_ok=True)
    indices = build_scope_indices(
        definition["dataset_size"], definition["scope_size"], definition["seed"]
    )
    arrays = {"indices": (output_root / "attribution_5k_indices.npy", indices)}
    if definition["mask"]:
        arrays["mask"] = (
            output_root / "attribution_5k_mask.npy",
            build_scope_mask(definition["dataset_size"], indices),
        )
    for path, value in arrays.values():
        if path.exists() and not force:
            existing = np.load(path)
            if not np.array_equal(existing, value):
                raise FileExistsError(f"Existing scope manifest differs: {path}")
        else:
            np.save(path, value)
    metadata = {
        "schema_version": 1,
        "experiment": experiment,
        "name": "paper_5k",
        "dataset_size": definition["dataset_size"],
        "scope_size": definition["scope_size"],
        "selection": "numpy.default_rng(seed).choice(replace=False)",
        "selection_seed": definition["seed"],
        "files": {name: {"path": path.name, "sha256": sha256(path)} for name, (path, _) in arrays.items()},
    }
    metadata_path = output_root / "attribution_scope.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create or verify a paper attribution scope")
    parser.add_argument("--experiment", required=True, choices=tuple(PAPER_SCOPES))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    output_root = args.output_root or repo_root() / "dataset_manifests" / args.experiment
    metadata = write_paper_scope(args.experiment, output_root, force=args.force)
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
