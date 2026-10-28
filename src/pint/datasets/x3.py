from __future__ import annotations

import argparse
import colorsys
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from tqdm.auto import tqdm

from .paths import data_root


SHAPES = (
    "big_square",
    "emptiness",
    "dot",
    "ring",
    "h",
    "horizontal_h",
    "cross",
    "x",
    "diamond",
)


def hue_label(hue: float) -> str:
    if hue >= 0.92 or hue < 0.08:
        return "red"
    for name, lower, upper in (
        ("orange", 0.08, 0.15),
        ("yellow", 0.15, 0.25),
        ("green", 0.25, 0.42),
        ("cyan", 0.42, 0.58),
        ("blue", 0.58, 0.75),
        ("purple", 0.75, 0.92),
    ):
        if lower <= hue < upper:
            return name
    return "red"


def shape_mask(name: str) -> np.ndarray:
    if name == "big_square":
        return np.ones((3, 3), dtype=bool)
    if name == "emptiness":
        return np.zeros((3, 3), dtype=bool)
    if name == "dot":
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, 1] = True
        return mask
    if name == "ring":
        mask = np.ones((3, 3), dtype=bool)
        mask[1, 1] = False
        return mask
    if name == "h":
        mask = np.ones((3, 3), dtype=bool)
        mask[0, 1] = mask[2, 1] = False
        return mask
    if name == "horizontal_h":
        mask = np.ones((3, 3), dtype=bool)
        mask[1, 0] = mask[1, 2] = False
        return mask
    if name == "cross":
        mask = np.zeros((3, 3), dtype=bool)
        mask[0, 1] = mask[1, 0] = mask[1, 1] = mask[1, 2] = mask[2, 1] = True
        return mask
    if name == "x":
        return np.eye(3, dtype=bool) | np.fliplr(np.eye(3, dtype=bool))
    if name == "diamond":
        mask = np.zeros((3, 3), dtype=bool)
        mask[0, 1] = mask[1, 0] = mask[1, 2] = mask[2, 1] = True
        return mask
    raise ValueError(f"Unknown x3 shape: {name}")


def generate_csv(path: Path, num_samples: int, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    fieldnames = ["id"] + [f"tile_{index}" for index in range(1, 10)] + [
        "label_1",
        "label_2",
        "label_3",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in tqdm(range(num_samples), desc="Preparing x3", unit="sample", dynamic_ncols=True):
            shape = str(rng.choice(SHAPES))
            shape_hue = float(rng.random_sample())
            background_hue = float(rng.random_sample())
            mask = shape_mask(shape).reshape(-1)
            # The original generator stores HSV-derived RGB values in a float32
            # grid, then converts each tile back to HSV while writing the CSV.
            # Preserve that round trip so the three-decimal CSV is byte-stable.
            shape_rgb = np.asarray(colorsys.hsv_to_rgb(shape_hue, 0.9, 0.9), dtype=np.float32)
            background_rgb = np.asarray(
                colorsys.hsv_to_rgb(background_hue, 0.9, 0.9), dtype=np.float32
            )
            shape_stored_hue = colorsys.rgb_to_hsv(*shape_rgb)[0]
            background_stored_hue = colorsys.rgb_to_hsv(*background_rgb)[0]
            tile_hues = np.where(mask, shape_stored_hue, background_stored_hue)
            labels = [
                f"shape_color_{hue_label(shape_hue)}",
                f"background_color_{hue_label(background_hue)}",
                f"shape_{shape}",
            ]
            order = np.random.RandomState(seed + index).permutation(3)
            row: dict[str, object] = {"id": index}
            row.update({f"tile_{i + 1}": f"{hue:.3f}" for i, hue in enumerate(tile_hues)})
            row.update({f"label_{i + 1}": labels[int(position)] for i, position in enumerate(order)})
            writer.writerow(row)


def main(argv: Sequence[str] | None = None) -> int:
    root = data_root()
    parser = argparse.ArgumentParser(description="Generate the fixed x3 training dataset")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--num-samples", type=int, default=50_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.out is None:
        args.out = root / "x3" / f"{args.seed}_{args.num_samples}.csv"
    if args.out.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {args.out}; pass --force to replace it")
    generate_csv(args.out, args.num_samples, args.seed)
    metadata = {
        "format_version": 1,
        "seed": args.seed,
        "size": args.num_samples,
        "csv": str(args.out.resolve()),
    }
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"saved {args.num_samples:,} x3 samples to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
