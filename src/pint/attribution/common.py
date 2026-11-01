from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from pint.query_gradients.common import atomic_savez


def open_timestamp_accumulator(
    output_dir: Path,
    shard_index: int,
    source: str,
    shape: tuple[int, int, int],
) -> tuple[Path, np.memmap]:
    directory = output_dir / "shards"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f".shard_{shard_index:04d}.{source}.timestamps.npy"
    values = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=shape
    )
    values.fill(0.0)
    return path, values


def finish_timestamp_accumulator(path: Path, values: np.memmap) -> np.ndarray:
    from tqdm.auto import tqdm

    scores = np.zeros(values.shape[1:], dtype=np.float64)
    for timestamp_slot in tqdm(
        range(values.shape[0]),
        desc=f"Reducing {path.name}",
        unit="timestamp",
        dynamic_ncols=True,
        leave=False,
    ):
        component = np.asarray(values[timestamp_slot], dtype=np.float64)
        scores += np.square(component)
    values.flush()
    values._mmap.close()
    os.unlink(path)
    return scores


def datapoint_shard(count: int, shard_index: int, shard_count: int) -> range:
    if count < 0:
        raise ValueError("datapoint count must be non-negative")
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid datapoint shard coordinates")
    quotient, remainder = divmod(count, shard_count)
    start = shard_index * quotient + min(shard_index, remainder)
    size = quotient + (1 if shard_index < remainder else 0)
    return range(start, start + size)


def load_query_parts(query_gradient_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = query_gradient_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = int(manifest["checkpoint_pairs"])
    parts = []
    for index in range(expected):
        path = query_gradient_dir / f"checkpoint_{index:04d}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            part = {name: np.asarray(payload[name]) for name in payload.files}
        if int(part["checkpoint_index"]) != index:
            raise ValueError(f"query-gradient checkpoint index mismatch: {path}")
        if not np.array_equal(part["timesteps"], np.asarray(manifest["timesteps"])):
            raise ValueError(f"query-gradient timestep mismatch: {path}")
        if int(part["projection_dim"]) != int(manifest["projection_dim"]):
            raise ValueError(f"query-gradient projection mismatch: {path}")
        parts.append(part)
    return manifest, parts


def shard_path(output_dir: Path, shard_index: int) -> Path:
    return output_dir / "shards" / f"shard_{shard_index:04d}.npz"


def save_shard(
    output_dir: Path,
    shard_index: int,
    *,
    raw_scores: np.ndarray,
    adamw_scores: np.ndarray | None,
    train_indices: np.ndarray,
    query_ids: np.ndarray,
    metadata: dict[str, Any],
) -> Path:
    path = shard_path(output_dir, shard_index)
    arrays = {
        "raw_scores": np.asarray(raw_scores, dtype=np.float64),
        "train_indices": np.asarray(train_indices, dtype=np.int64),
        "query_ids": np.asarray(query_ids, dtype=np.int32),
        "manifest_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    if adamw_scores is not None:
        arrays["adamw_scores"] = np.asarray(adamw_scores, dtype=np.float64)
    atomic_savez(path, **arrays)
    return path


def merge_shards(output_dir: Path, shard_count: int) -> Path:
    records = []
    for index in range(shard_count):
        path = shard_path(output_dir, index)
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            records.append(
                {
                    "path": path,
                    "raw_scores": np.asarray(payload["raw_scores"], dtype=np.float64),
                    "adamw_scores": (
                        np.asarray(payload["adamw_scores"], dtype=np.float64)
                        if "adamw_scores" in payload.files else None
                    ),
                    "indices": np.asarray(payload["train_indices"], dtype=np.int64),
                    "query_ids": np.asarray(payload["query_ids"], dtype=np.int32),
                    "manifest": json.loads(str(np.asarray(payload["manifest_json"]).item())),
                }
            )
    reference = records[0]
    immutable = (
        "format_version", "experiment", "method", "normalization",
        "score_normalization",
        "projection", "projection_dim", "timesteps", "checkpoint_pairs",
        "query_manifest_sha256", "scope", "model_seed", "mc_samples",
        "mc_pairing", "reduction", "reduction_formula", "learning_rate_weighted",
        "parameter_sources", "adamw_enabled", "adamw_semantics",
        "adamw_score_multiplier",
    )
    for record in records:
        if not np.array_equal(record["query_ids"], reference["query_ids"]):
            raise ValueError(f"query IDs differ in {record['path']}")
        shape = (len(reference["query_ids"]), len(record["indices"]))
        if record["raw_scores"].shape != shape:
            raise ValueError(f"invalid attribution shard shape: {record['path']}")
        if (record["adamw_scores"] is None) != (reference["adamw_scores"] is None):
            raise ValueError(f"AdamW output differs in {record['path']}")
        if record["adamw_scores"] is not None and record["adamw_scores"].shape != shape:
            raise ValueError(f"invalid AdamW attribution shard shape: {record['path']}")
        mismatches = [
            key for key in immutable
            if record["manifest"].get(key) != reference["manifest"].get(key)
        ]
        if mismatches:
            raise ValueError(f"incompatible attribution shard {record['path']}: {mismatches}")
    indices = np.concatenate([record["indices"] for record in records])
    if len(np.unique(indices)) != len(indices):
        raise ValueError("attribution shards contain duplicate datapoints")
    expected = int(reference["manifest"]["scope_size"])
    if len(indices) != expected:
        raise ValueError(f"merged {len(indices)} datapoints; expected {expected}")
    raw_scores = np.concatenate([record["raw_scores"] for record in records], axis=1)
    adamw_scores = (
        np.concatenate([record["adamw_scores"] for record in records], axis=1)
        if reference["adamw_scores"] is not None else None
    )
    metadata = dict(reference["manifest"])
    metadata["merged_shards"] = [str(record["path"]) for record in records]
    metadata["shard_count"] = shard_count
    output = output_dir / "scores.npz"
    arrays = {
        "raw_scores": raw_scores,
        "train_indices": indices,
        "query_ids": reference["query_ids"],
        "manifest_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    if adamw_scores is not None:
        arrays["adamw_scores"] = adamw_scores
    atomic_savez(output, **arrays)
    return output


def attribution_metadata(
    *,
    experiment: str,
    query_manifest: Path,
    query_gradient_manifest: dict[str, Any],
    scope: str,
    scope_size: int,
    model_seed: int,
    shard_index: int,
    shard_count: int,
    batch_size: int,
    adamw_enabled: bool,
) -> dict[str, Any]:
    query_digest = hashlib.sha256(query_manifest.read_bytes()).hexdigest()
    expected_digest = query_gradient_manifest.get("query_manifest_sha256")
    if expected_digest is not None and expected_digest != query_digest:
        raise RuntimeError(
            "query manifest does not match the manifest used to compute query gradients"
        )
    return {
        "format_version": 2,
        "experiment": experiment,
        "method": "pint",
        "normalization": query_gradient_manifest["normalization"],
        "score_normalization": "raw" if experiment == "x3" else "query_train_l2",
        "projection": query_gradient_manifest["projection"],
        "projection_dim": int(query_gradient_manifest["projection_dim"]),
        "timesteps": list(query_gradient_manifest["timesteps"]),
        "checkpoint_pairs": int(query_gradient_manifest["checkpoint_pairs"]),
        "query_manifest_sha256": query_digest,
        "scope": scope,
        "scope_size": scope_size,
        "model_seed": model_seed,
        "mc_samples": 1,
        "mc_pairing": "shared_with_query_by_checkpoint_and_timestep",
        "reduction": "timestamp_sum_squared",
        "reduction_formula": "sum_t((sum_checkpoint learning_rate * dot / num_timesteps)^2)",
        "learning_rate_weighted": True,
        "parameter_sources": ["raw_gradient", "adamw_full"] if adamw_enabled else ["raw_gradient"],
        "adamw_enabled": adamw_enabled,
        "adamw_semantics": (
            "next full AdamW direction from saved moments; learning rate applied once"
            if adamw_enabled else None
        ),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "batch_size": batch_size,
    }
