from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def normalization_for(experiment: str) -> str:
    return "predicted_noise_delta_l2"


def timestep_count_for(experiment: str) -> int:
    return 100 if experiment in ("x3", "shape3d") else 50


def timestep_grid(experiment: str, total: int = 1000) -> np.ndarray:
    count = timestep_count_for(experiment)
    if experiment == "x3":
        # Match torch.linspace(T-1, 0, 100).long() and the later x3
        # TracIn-DAS timestamp bank exactly.
        return np.linspace(total - 1, 0, count, dtype=np.int32)
    if experiment == "shape3d":
        positions = np.linspace(0, total - 1, count, dtype=np.int32)
        return (total - 1 - positions).astype(np.int32)
    if total % count:
        raise ValueError(
            f"{experiment} sampling requires {count} steps to divide {total} timesteps"
        )
    return np.arange(total - 1, -1, -(total // count), dtype=np.int32)


def validate_sampled_timesteps(
    query_dir: Path, expected: np.ndarray, *, query_id: int
) -> None:
    path = query_dir / "trajectory_t.npy"
    actual = np.load(path)
    if not np.array_equal(actual, expected):
        raise RuntimeError(
            f"Query {query_id} sampling timesteps do not match query-gradient "
            f"timesteps; rerun 02_sample_queries.py with the default settings"
        )


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def stable_torch_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def x3_tracin_das_noise_seed(
    checkpoint_index: int, timestamp_index: int, timestep: int
) -> int:
    return stable_torch_seed(
        7367,
        "tracin_das_checkpoint_timestamp_shared_noise",
        checkpoint_index,
        timestamp_index,
        timestep,
    )


def read_query_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        metadata: dict[str, Any] = {}
        queries = payload
    elif isinstance(payload, dict) and isinstance(payload.get("queries"), list):
        metadata = {key: value for key, value in payload.items() if key != "queries"}
        queries = payload["queries"]
    else:
        raise ValueError(f"Invalid query manifest: {path}")
    if not queries:
        raise ValueError(f"Query manifest contains no queries: {path}")
    ids = [int(query["query_id"]) for query in queries]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Query IDs are not unique in {path}")
    return metadata, queries


def batched(values: list[Any], size: int) -> Iterable[list[Any]]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def atomic_savez(path: Path, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def part_path(output_dir: Path, index: int) -> Path:
    return output_dir / f"checkpoint_{index:04d}.npz"


def checkpoint_pair_range(
    pair_count: int, shard_index: int = 0, shard_count: int = 1
) -> range:
    """Return one balanced, contiguous checkpoint-pair shard."""
    if pair_count < 0:
        raise ValueError("checkpoint pair count must be non-negative")
    if shard_count <= 0:
        raise ValueError("shard count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard index must be between zero and shard count minus one")
    quotient, remainder = divmod(pair_count, shard_count)
    start = shard_index * quotient + min(shard_index, remainder)
    size = quotient + (1 if shard_index < remainder else 0)
    return range(start, start + size)


def write_run_manifest(
    output_dir: Path,
    *,
    experiment: str,
    query_manifest: Path,
    checkpoints: list[Path],
    timesteps: np.ndarray,
    projection_dim: int,
    batch_size: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    query_manifest_digest = hashlib.sha256(query_manifest.read_bytes()).hexdigest()
    payload = {
        "format_version": 1,
        "experiment": experiment,
        "method": "pint",
        "objective": "predicted_noise_next_checkpoint_delta",
        "normalization": normalization_for(experiment),
        "checkpoint_order": [str(path) for path in checkpoints],
        "checkpoint_pairs": max(0, len(checkpoints) - 1),
        "query_manifest": str(query_manifest),
        "query_manifest_sha256": query_manifest_digest,
        "timesteps": timesteps.tolist(),
        "timestep_count": int(len(timesteps)),
        "projection": "countsketch",
        "projection_dim": projection_dim,
        "query_batch_size": batch_size,
        "checkpoint_major": True,
    }
    manifest_path = output_dir / "manifest.json"
    immutable_keys = (
        "experiment",
        "method",
        "objective",
        "normalization",
        "checkpoint_order",
        "query_manifest_sha256",
        "timesteps",
        "projection",
        "projection_dim",
    )
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = [
            key for key in immutable_keys if existing.get(key) != payload.get(key)
        ]
        if mismatches:
            raise RuntimeError(
                f"Refusing to mix incompatible query gradients in {output_dir}; "
                f"changed fields: {', '.join(mismatches)}"
            )
        return
    contents = json.dumps(payload, indent=2) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=".manifest.", suffix=".json", dir=output_dir
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(contents)
        try:
            os.link(temporary, manifest_path)
        except FileExistsError:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            mismatches = [
                key for key in immutable_keys if existing.get(key) != payload.get(key)
            ]
            if mismatches:
                raise RuntimeError(
                    f"Refusing to mix incompatible query gradients in {output_dir}; "
                    f"changed fields: {', '.join(mismatches)}"
                )
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
