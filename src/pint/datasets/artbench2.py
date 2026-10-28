from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from .paths import data_root


CLASSES = ("post_impressionism", "ukiyo_e")


def save_pickle(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def main(argv: Sequence[str] | None = None) -> int:
    root = data_root()
    parser = argparse.ArgumentParser(description="Prepare the fixed ArtBench-2 split")
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=root / "raw" / "artbench2" / "ArtBench-10.csv",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=root / "raw" / "artbench2" / "artbench-10-imagefolder",
    )
    parser.add_argument("--out-root", type=Path, default=root / "artbench2" / "indices")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=5_000)
    parser.add_argument("--validation-size", type=int, default=1_000)
    parser.add_argument("--skip-image-check", action="store_true")
    args = parser.parse_args(argv)
    if not args.metadata_csv.is_file():
        raise FileNotFoundError(args.metadata_csv)
    frame = pd.read_csv(args.metadata_csv)
    missing_columns = {"name", "label", "split"}.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Metadata is missing columns: {sorted(missing_columns)}")
    frame = frame.copy()
    frame["image_path"] = frame.apply(
        lambda row: str(args.image_root / row["label"] / row["name"]), axis=1
    )
    frame["prompt"] = frame["label"].map(
        lambda label: f"a {str(label).replace('_', ' ').lower()} painting"
    )
    selected = frame[frame["label"].isin(CLASSES)].copy()
    train_pool = selected[selected["split"] == "train"]
    validation_pool = selected[selected["split"] == "test"]
    if len(train_pool) < args.train_size:
        raise ValueError(
            f"ArtBench-2 train pool has {len(train_pool)} rows; need {args.train_size}"
        )
    if len(validation_pool) < args.validation_size:
        raise ValueError(
            f"ArtBench-2 test pool has {len(validation_pool)} rows; "
            f"need {args.validation_size}"
        )
    train, _ = train_test_split(
        train_pool,
        train_size=args.train_size,
        random_state=args.seed,
        stratify=train_pool["label"],
    )
    validation, _ = train_test_split(
        validation_pool,
        train_size=args.validation_size,
        random_state=args.seed,
        stratify=validation_pool["label"],
    )
    if not args.skip_image_check:
        rows = pd.concat([train, validation])
        for row in tqdm(rows.itertuples(), total=len(rows), desc="Checking ArtBench-2 images", unit="image"):
            path = args.image_root / row.label / row.name
            if not path.is_file():
                raise FileNotFoundError(path)
    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest_root = args.out_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    save_pickle(args.out_root / "idx-train.pkl", train.index.tolist())
    save_pickle(args.out_root / "idx-val.pkl", validation.index.tolist())
    train.to_csv(manifest_root / "train.csv", index_label="source_index")
    validation.to_csv(manifest_root / "validation.csv", index_label="source_index")
    summary = {
        "classes": list(CLASSES),
        "candidate_train_rows": len(train_pool),
        "selected_train_rows": len(train),
        "selected_validation_rows": len(validation),
        "split_seed": args.seed,
        "train_class_counts": train["label"].value_counts().sort_index().to_dict(),
        "validation_class_counts": validation["label"].value_counts().sort_index().to_dict(),
    }
    (args.out_root / "split_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"saved ArtBench-2 split to {args.out_root}")
    print(f"train={len(train):,} validation={len(validation):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
