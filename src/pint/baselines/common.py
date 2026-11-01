from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


BASELINE_METHODS = ("gas", "tracincp", "retrac")


def orient_adamw_scores(method: str, scores: np.ndarray | None) -> np.ndarray | None:
    if method not in BASELINE_METHODS:
        raise ValueError(f"unknown baseline method: {method}")
    return None if scores is None else -np.asarray(scores)


@dataclass(frozen=True)
class BaselineSpec:
    method: str
    experiment: str
    timestep_count: int
    mc_samples: int
    query_reduction: str
    score_normalization: str


def baseline_spec(method: str, experiment: str) -> BaselineSpec:
    if method not in BASELINE_METHODS:
        raise ValueError(f"unknown baseline method: {method}")
    if experiment not in ("x3", "shape3d", "celeba", "artbench2"):
        raise ValueError(f"unknown experiment: {experiment}")
    timestep_count = 100 if experiment in ("x3", "shape3d") else 50
    mc_samples = 10 if experiment == "x3" else 1
    if method == "retrac":
        query_reduction = "mean_projected_full_l2_per_loss"
        score_normalization = "full_l2_before_projection"
    elif experiment == "x3":
        query_reduction = "timewise_mean_mc"
        score_normalization = "projected_both_l2" if method == "gas" else "raw"
    else:
        query_reduction = "gradient_of_mean_loss"
        score_normalization = "projected_both_l2" if method == "gas" else "raw"
    return BaselineSpec(
        method=method,
        experiment=experiment,
        timestep_count=timestep_count,
        mc_samples=mc_samples,
        query_reduction=query_reduction,
        score_normalization=score_normalization,
    )


def run_tag(spec: BaselineSpec, projection_dim: int) -> str:
    if projection_dim <= 0:
        raise ValueError("projection_dim must be positive")
    return (
        f"{spec.method}_t{spec.timestep_count}_mc{spec.mc_samples}_"
        f"projected{projection_dim}"
    )


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def query_part_path(output_dir: Path, checkpoint_index: int, tag: str) -> Path:
    return output_dir / f"checkpoint_{checkpoint_index:04d}__{tag}.npz"


def atomic_savez(path: Path, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **values)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_query_manifest(
    output_dir: Path,
    *,
    spec: BaselineSpec,
    tag: str,
    projection_dim: int,
    query_manifest: Path,
    checkpoints: list[Path],
    source_checkpoint_count: int | None = None,
) -> dict[str, Any]:
    if source_checkpoint_count is None:
        source_checkpoint_count = len(checkpoints)
    payload = {
        "format_version": 1,
        "kind": "baseline_query_features",
        "tag": tag,
        **asdict(spec),
        "projection": "countsketch",
        "projection_dim": int(projection_dim),
        "query_manifest": str(query_manifest.resolve()),
        "query_manifest_sha256": hashlib.sha256(query_manifest.read_bytes()).hexdigest(),
        "checkpoint_order": [str(path.resolve()) for path in checkpoints],
        "source_checkpoint_count": int(source_checkpoint_count),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"manifest__{tag}.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"refusing to mix incompatible baseline query runs in {output_dir}")
    else:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".json", dir=output_dir
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, indent=2) + "\n")
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return payload


def load_query_run(output_dir: Path, tag: str) -> tuple[dict[str, Any], list[dict[str, np.ndarray]]]:
    manifest_path = output_dir / f"manifest__{tag}.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parts = []
    for checkpoint_index in range(int(manifest["source_checkpoint_count"])):
        path = query_part_path(output_dir, checkpoint_index, tag)
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            parts.append({key: np.asarray(payload[key]) for key in payload.files})
    return manifest, parts


def validate_query_source(manifest: dict[str, Any], query_manifest: Path) -> None:
    digest = hashlib.sha256(query_manifest.read_bytes()).hexdigest()
    if digest != manifest.get("query_manifest_sha256"):
        raise RuntimeError(
            "query manifest differs from the bank used to compute baseline query features"
        )
