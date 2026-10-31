from __future__ import annotations

import math
import random
from typing import Mapping, Sequence

import numpy as np


SHAPE3D_LABEL_GROUPS = (
    tuple(f"shape_{name}" for name in ("cube", "cylinder", "sphere", "capsule")),
    tuple(f"object_hue_{index}" for index in range(10)),
    tuple(f"wall_hue_{index}" for index in range(10)),
    tuple(f"floor_hue_{index}" for index in range(10)),
)
ARTBENCH2_LABELS = ("post_impressionism", "ukiyo_e")
X3_PROMPT_SEED = 20260818


def default_query_count(experiment: str) -> int:
    return 60 if experiment == "artbench2" else 100


def default_query_seed(experiment: str) -> int:
    return 0


def shape3d_queries(count: int, seed: int = 0) -> list[dict[str, object]]:
    """Build one-per-category queries using the Shape3D LDS prompt contract."""

    records = []
    for query_id in range(count):
        query_seed = seed + query_id
        rng = np.random.default_rng(query_seed)
        labels = [str(rng.choice(group)) for group in SHAPE3D_LABEL_GROUPS]
        records.append(
            {
                "query_id": query_id,
                "query_seed": query_seed,
                "initial_seed": query_seed,
                "family": "prompted",
                "labels": labels,
                "prompt": ",".join(labels),
            }
        )
    return records


def celeba_queries(
    count: int,
    attribute_names: Sequence[str],
    attributes: Mapping[str, np.ndarray],
    partitions: Mapping[str, int],
    seed: int = 0,
) -> list[dict[str, object]]:
    """Reproduce the prompted Attr40 validation-label query bank."""

    blurry_index = list(attribute_names).index("Blurry")
    candidates = []
    for filename in sorted(attributes):
        if partitions.get(filename) != 1:
            continue
        labels = (np.asarray(attributes[filename]) > 0).astype(np.uint8)
        positive_count = int(labels.sum())
        if labels[blurry_index] or not 4 <= positive_count <= 14:
            continue
        candidates.append((filename, labels))
    prompt_count = math.ceil(count / 4)
    if prompt_count > len(candidates):
        raise ValueError(
            f"requested {count} queries but only {len(candidates)} CelebA prompts are eligible"
        )
    positions = np.random.default_rng(42).choice(
        len(candidates), size=prompt_count, replace=False
    )
    records = []
    for prompt_id, position in enumerate(positions.tolist()):
        filename, labels = candidates[position]
        positive_names = [
            name for name, value in zip(attribute_names, labels.tolist()) if value
        ]
        for seed_offset in range(4):
            if len(records) == count:
                return records
            initial_seed = seed + seed_offset
            records.append(
                {
                    "query_id": len(records),
                    "query_seed": initial_seed,
                    "initial_seed": initial_seed,
                    "prompt_id": prompt_id,
                    "family": "prompted",
                    "source_partition": 1,
                    "source_filename": filename,
                    "selection_seed": 42,
                    "labels": labels.astype(int).tolist(),
                    "positive_attributes": positive_names,
                    "prompt": ",".join(positive_names),
                }
            )
    return records


def artbench2_queries(count: int, seed: int = 0) -> list[dict[str, object]]:
    records = []
    for query_id in range(count):
        label = ARTBENCH2_LABELS[query_id % len(ARTBENCH2_LABELS)]
        query_seed = seed + query_id // len(ARTBENCH2_LABELS)
        records.append(
            {
                "query_id": query_id,
                "query_seed": query_seed,
                "initial_seed": query_seed,
                "family": "prompted",
                "label": label,
                "labels": [label],
                "prompt": f"a {label.replace('_', ' ')} painting",
            }
        )
    return records


def _x3_prompt_groups(vocabulary: Sequence[str]) -> dict[str, list[str]]:
    groups = {
        "shape_color": [label for label in vocabulary if label.startswith("shape_color_")],
        "background_color": [
            label for label in vocabulary if label.startswith("background_color_")
        ],
        "shape": [
            label
            for label in vocabulary
            if label.startswith("shape_") and not label.startswith("shape_color_")
        ],
    }
    if any(not values for values in groups.values()):
        raise ValueError(f"Could not form x3 prompt groups from vocabulary: {groups}")
    return groups


def x3_queries(
    count: int, vocabulary: Sequence[str], seed: int = 0
) -> list[dict[str, object]]:
    """Build the prompted-only x3 query bank."""

    groups = _x3_prompt_groups(vocabulary)
    rng = random.Random(X3_PROMPT_SEED)
    records = []
    for query_id in range(count):
        labels = [
            rng.choice(groups["shape_color"]),
            rng.choice(groups["background_color"]),
            rng.choice(groups["shape"]),
        ]
        records.append(
            {
                "query_id": query_id,
                "query_seed": seed + query_id,
                "initial_seed": seed + query_id,
                "family": "prompted",
                "labels": labels,
                "prompt": ",".join(labels),
            }
        )
    return records
