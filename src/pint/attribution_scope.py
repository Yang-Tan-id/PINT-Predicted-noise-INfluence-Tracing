from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_yaml, repo_root, scope_path


@dataclass(frozen=True)
class AttributionScope:
    name: str
    selection: str
    expected_size: int | None = None
    indices_path: Path | None = None


def load_scope(
    experiment: str,
    name: str,
    root: Path | None = None,
) -> AttributionScope:
    base = root.resolve() if root is not None else repo_root()
    payload = load_yaml(scope_path(experiment, name, base))
    selection = payload.get("selection")
    if selection not in {"all", "explicit_indices"}:
        raise ValueError(f"Unsupported attribution scope selection: {selection!r}")
    configured_path = payload.get("indices")
    indices_path = None
    if configured_path is not None:
        indices_path = (base / configured_path).resolve()
    expected_size = payload.get("expected_size")
    if expected_size is not None and (not isinstance(expected_size, int) or expected_size <= 0):
        raise ValueError("expected_size must be a positive integer or null")
    return AttributionScope(
        name=str(payload.get("name", name)),
        selection=selection,
        expected_size=expected_size,
        indices_path=indices_path,
    )


def resolve_indices(scope: AttributionScope, dataset_size: int) -> np.ndarray:
    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive")
    if scope.selection == "all":
        indices = np.arange(dataset_size, dtype=np.int64)
    else:
        if scope.indices_path is None:
            raise ValueError("explicit_indices scope requires an indices path")
        if not scope.indices_path.is_file():
            raise FileNotFoundError(scope.indices_path)
        indices = np.asarray(np.load(scope.indices_path), dtype=np.int64).reshape(-1)
    if indices.size == 0:
        raise ValueError("Attribution scope is empty")
    if np.unique(indices).size != indices.size:
        raise ValueError("Attribution scope contains duplicate indices")
    if indices.min() < 0 or indices.max() >= dataset_size:
        raise ValueError(
            f"Attribution indices must be within [0, {dataset_size - 1}]"
        )
    if scope.expected_size is not None and indices.size != scope.expected_size:
        raise ValueError(
            f"Scope {scope.name!r} expected {scope.expected_size} indices, got {indices.size}"
        )
    return indices


def scope_summary(scope: AttributionScope) -> dict[str, Any]:
    return {
        "name": scope.name,
        "selection": scope.selection,
        "expected_size": scope.expected_size,
        "indices": None if scope.indices_path is None else str(scope.indices_path),
    }
