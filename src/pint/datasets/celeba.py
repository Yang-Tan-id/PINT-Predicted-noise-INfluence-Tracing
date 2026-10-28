from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path, PurePosixPath
from typing import Sequence

import numpy as np


def read_partition(path: Path) -> dict[str, int]:
    partitions: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 2:
                raise ValueError(f"Malformed partition row {line_number}: {line.rstrip()!r}")
            filename, split_text = fields
            split = int(split_text)
            if split not in (0, 1, 2):
                raise ValueError(f"Invalid split {split} on row {line_number}")
            if filename in partitions:
                raise ValueError(f"Duplicate partition filename: {filename}")
            partitions[filename] = split
    if not partitions:
        raise ValueError(f"No partition rows found in {path}")
    return partitions


def read_attributes(path: Path) -> tuple[tuple[str, ...], dict[str, np.ndarray]]:
    with path.open("r", encoding="utf-8") as handle:
        declared_count = int(handle.readline().strip())
        names = tuple(handle.readline().split())
        if len(names) != 40:
            raise ValueError(f"Expected 40 CelebA attributes, found {len(names)}")
        attributes: dict[str, np.ndarray] = {}
        for line_number, line in enumerate(handle, start=3):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 41:
                raise ValueError(f"Malformed attribute row {line_number}")
            values = np.asarray([int(value) for value in fields[1:]], dtype=np.int8)
            if not np.all(np.isin(values, (-1, 1))):
                raise ValueError(f"Attribute row {line_number} contains invalid values")
            if fields[0] in attributes:
                raise ValueError(f"Duplicate attribute filename: {fields[0]}")
            attributes[fields[0]] = values
    if len(attributes) != declared_count:
        raise ValueError(f"Expected {declared_count:,} attribute rows, found {len(attributes):,}")
    return names, attributes


def select_training_filenames(
    partitions: dict[str, int], seed: int, train_size: int
) -> np.ndarray:
    training = np.asarray(
        sorted(filename for filename, split in partitions.items() if split == 0), dtype="U32"
    )
    if train_size <= 0 or train_size > len(training):
        raise ValueError(f"train_size must be within [1, {len(training)}]")
    chosen = np.random.default_rng(seed).choice(len(training), size=train_size, replace=False)
    return training[chosen]


def center_crop_resize(image, crop_size: int, image_size: int):
    from PIL import Image

    image = image.convert("RGB")
    width, height = image.size
    if width < crop_size or height < crop_size:
        raise ValueError(f"Image {width}x{height} is smaller than the requested crop")
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    image = image.crop((left, top, left + crop_size, top + crop_size))
    return image.resize((image_size, image_size), resample=Image.Resampling.BICUBIC)


def zip_members_by_basename(archive: zipfile.ZipFile) -> dict[str, str]:
    members: dict[str, str] = {}
    for info in archive.infolist():
        basename = PurePosixPath(info.filename).name
        if info.is_dir() or not basename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        if basename in members:
            raise ValueError(f"Archive contains duplicate image basename: {basename}")
        members[basename] = info.filename
    return members


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare the fixed 10k CelebA dataset")
    parser.add_argument("--images-zip", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--attributes", type=Path)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=10_000)
    parser.add_argument("--crop-size", type=int, default=140)
    parser.add_argument("--image-size", type=int, default=64)
    args = parser.parse_args(argv)
    if args.crop_size <= 0 or args.image_size <= 0:
        parser.error("--crop-size and --image-size must be positive")
    try:
        from PIL import Image
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise SystemExit("CelebA preparation requires pillow and tqdm") from exc

    partitions = read_partition(args.partition)
    filenames = select_training_filenames(partitions, args.seed, args.train_size)
    if args.attributes is None:
        label_names = np.asarray(["face"])
        labels = np.ones((args.train_size, 1), dtype=np.uint8)
        raw_attributes = None
    else:
        attribute_names, rows = read_attributes(args.attributes)
        missing = [name for name in filenames if str(name) not in rows]
        if missing:
            raise FileNotFoundError(f"Attribute file is missing {missing[0]}")
        raw_attributes = np.stack([rows[str(name)] for name in filenames]).astype(np.int8)
        labels = (raw_attributes > 0).astype(np.uint8)
        label_names = np.asarray(attribute_names)
    images = np.empty((args.train_size, args.image_size, args.image_size, 3), dtype=np.uint8)
    with zipfile.ZipFile(args.images_zip, "r") as archive:
        members = zip_members_by_basename(archive)
        missing = [name for name in filenames if name not in members]
        if missing:
            raise FileNotFoundError(f"Image archive is missing {missing[0]}")
        for index, filename in enumerate(tqdm(filenames, desc="Preparing CelebA", unit="image")):
            with archive.open(members[str(filename)]) as handle:
                with Image.open(io.BytesIO(handle.read())) as image:
                    images[index] = np.asarray(
                        center_crop_resize(image, args.crop_size, args.image_size), dtype=np.uint8
                    )

    args.out_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "images": images,
        "labels": labels,
        "source_filenames": filenames,
        "label_names": label_names,
    }
    if raw_attributes is not None:
        payload["raw_attributes"] = raw_attributes
    np.savez_compressed(args.out_root / "dataset.npz", **payload)
    np.save(args.out_root / "celeba_10k_source_filenames.npy", filenames)
    metadata = {
        "format_version": 1,
        "source_images_zip": str(args.images_zip.resolve()),
        "source_partition": str(args.partition.resolve()),
        "source_split": "train (partition 0)",
        "selection_seed": args.seed,
        "size": args.train_size,
        "crop": f"center {args.crop_size}x{args.crop_size}",
        "resize": f"{args.image_size}x{args.image_size} bicubic",
        "conditioning": "unconditional" if args.attributes is None else "40-way multi-hot positive CelebA attributes",
        "source_attributes": None if args.attributes is None else str(args.attributes.resolve()),
        "attribute_names": label_names.tolist(),
        "attribution_scope": "managed separately under dataset_manifests/celeba",
    }
    (args.out_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved {args.train_size:,} images to {args.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
