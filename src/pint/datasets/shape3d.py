from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np


SHAPE_NAMES = ("cube", "cylinder", "sphere", "capsule")
GROUP_COLUMNS = (0, 1, 2, 4)


def factor_ids(labels: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    ids = np.empty(labels.shape, dtype=np.int16)
    values: list[np.ndarray] = []
    for column in range(labels.shape[1]):
        unique, inverse = np.unique(labels[:, column], return_inverse=True)
        values.append(unique)
        ids[:, column] = inverse.astype(np.int16)
    return ids, values


def select_balanced_indices(
    labels: np.ndarray,
    seed: int,
    samples_per_group: int,
    *,
    show_progress: bool = False,
) -> np.ndarray:
    ids, values = factor_ids(labels)
    expected = (10, 10, 10, 8, 4, 15)
    actual = tuple(len(value) for value in values)
    if actual != expected:
        raise ValueError(f"Unexpected factor cardinalities {actual}; expected {expected}")
    group_id = np.ravel_multi_index(
        tuple(ids[:, column] for column in GROUP_COLUMNS), dims=(10, 10, 10, 4)
    )
    order = np.argsort(group_id, kind="stable")
    counts = np.bincount(group_id, minlength=4000)
    if not np.all(counts == 120):
        bad = np.flatnonzero(counts != 120)
        raise ValueError(f"Expected 120 examples in every group; bad groups: {bad[:10].tolist()}")
    offsets = np.concatenate(([0], np.cumsum(counts)))
    groups = range(4000)
    if show_progress:
        from tqdm.auto import tqdm

        groups = tqdm(groups, total=4000, desc="Selecting balanced groups", unit="group")
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for group in groups:
        candidates = order[offsets[group] : offsets[group + 1]]
        selected.extend(rng.choice(candidates, size=samples_per_group, replace=False).tolist())
    return np.asarray(selected, dtype=np.int64)


def build_conditions(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    names = (
        tuple(f"shape_{name}" for name in SHAPE_NAMES)
        + tuple(f"object_hue_{index}" for index in range(10))
        + tuple(f"wall_hue_{index}" for index in range(10))
        + tuple(f"floor_hue_{index}" for index in range(10))
    )
    labels = np.zeros((len(ids), len(names)), dtype=np.uint8)
    rows = np.arange(len(ids))
    labels[rows, ids[:, 4]] = 1
    labels[rows, 4 + ids[:, 2]] = 1
    labels[rows, 14 + ids[:, 1]] = 1
    labels[rows, 24 + ids[:, 0]] = 1
    return labels, np.asarray(names)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the balanced 20k Shape3D dataset")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-group", type=int, default=5)
    parser.add_argument("--read-chunk-size", type=int, default=256)
    args = parser.parse_args(argv)
    if args.read_chunk_size <= 0:
        parser.error("--read-chunk-size must be positive")
    try:
        import h5py
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise SystemExit("Shape3D preparation requires h5py and tqdm") from exc

    with h5py.File(args.input, "r") as source:
        labels_source = source["labels"]
        labels_all = np.empty(labels_source.shape, dtype=np.float64)
        label_chunk = max(1, args.read_chunk_size * 64)
        for start in tqdm(range(0, len(labels_source), label_chunk), desc="Reading labels"):
            end = min(start + label_chunk, len(labels_source))
            labels_all[start:end] = labels_source[start:end]
        selected = select_balanced_indices(
            labels_all, args.seed, args.samples_per_group, show_progress=True
        )
        sort_order = np.argsort(selected)
        sorted_indices = selected[sort_order]
        images = np.empty((len(selected),) + tuple(source["images"].shape[1:]), dtype=np.uint8)
        for start in tqdm(range(0, len(selected), args.read_chunk_size), desc="Reading images"):
            end = min(start + args.read_chunk_size, len(selected))
            images[sort_order[start:end]] = source["images"][sorted_indices[start:end]]

    raw_labels = labels_all[selected]
    if len(selected) != 4000 * args.samples_per_group:
        raise AssertionError("balanced selection has the wrong size")
    ids, values = factor_ids(raw_labels)
    conditions, label_names = build_conditions(ids)
    args.out_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_root / "dataset.npz",
        images=images,
        labels=conditions,
        raw_labels=raw_labels,
        factor_ids=ids,
        source_indices=selected,
        label_names=label_names,
    )
    np.save(args.out_root / "3dshapes_20k_source_indices.npy", selected)
    metadata = {
        "format_version": 1,
        "source": str(args.input.resolve()),
        "selection_seed": args.seed,
        "samples_per_group": args.samples_per_group,
        "num_groups": 4000,
        "size": len(selected),
        "conditioning": "unordered 34-way multi-hot set",
        "condition_labels": label_names.tolist(),
        "factor_order": ["floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation"],
        "factor_values": [value.tolist() for value in values],
        "attribution_scope": "managed separately under dataset_manifests/shape3d",
    }
    (args.out_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved {len(selected):,} samples to {args.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
