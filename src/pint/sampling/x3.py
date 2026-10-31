from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from pint.sampling.queries import x3_queries


def _final_checkpoint(model_dir: Path, family: str) -> Path:
    paths = sorted((model_dir / "base" / family).glob("epoch_*.pt"))
    if not paths:
        raise FileNotFoundError(
            f"No {family} checkpoint found under {model_dir}; run 01_train_model.py first"
        )
    return paths[-1]


def _model_seed(model_dir: Path, family: str, checkpoint: dict) -> int:
    checkpoint_seed = checkpoint.get("seed")
    meta_path = model_dir / "base" / family / "meta.json"
    metadata_seed = None
    if meta_path.is_file():
        metadata_seed = json.loads(meta_path.read_text(encoding="utf-8")).get("seed")
    known = {int(value) for value in (checkpoint_seed, metadata_seed) if value is not None}
    if not known:
        raise RuntimeError(
            f"Missing model seed metadata for x3 {family}; retrain with the reproducible entrypoint"
        )
    if len(known) != 1:
        raise RuntimeError(
            f"Conflicting x3 model seeds for {family}: {sorted(known)}"
        )
    return known.pop()


def _save_preview(array: np.ndarray, path: Path) -> None:
    from PIL import Image

    image = np.clip(array, 0.0, 1.0)
    image = np.transpose(image, (1, 2, 0))
    Image.fromarray(np.rint(image * 255).astype(np.uint8)).resize(
        (192, 192), resample=Image.Resampling.NEAREST
    ).save(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sample x3 attribution queries")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ddim-steps", type=int, default=1000)
    parser.add_argument("--trajectory-snapshots", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    args = parser.parse_args(argv)
    if args.num_queries <= 0:
        parser.error("--num-queries must be positive")
    if not 1 <= args.trajectory_snapshots <= args.ddim_steps:
        parser.error("--trajectory-snapshots must be between 1 and --ddim-steps")

    try:
        import torch
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise SystemExit("x3 sampling requires torch, Pillow, and tqdm") from exc

    from pint.training.backends import x3_model

    if args.device == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("--device gpu requested, but CUDA is unavailable")
    device = torch.device(
        "cuda:0" if args.device != "cpu" and torch.cuda.is_available() else "cpu"
    )
    checkpoints = {
        "prompted": torch.load(
            _final_checkpoint(args.model_dir, "prompted"), map_location="cpu"
        )
    }
    model_seed = _model_seed(args.model_dir, "prompted", checkpoints["prompted"])
    vocabulary = [
        label
        for label, _ in sorted(checkpoints["prompted"]["vocab"].items(), key=lambda item: item[1])
    ]
    records = x3_queries(args.num_queries, vocabulary, args.seed)
    models = {}
    for family, checkpoint in checkpoints.items():
        config = checkpoint.get("config", {})
        model = x3_model.CondEpsModel(
            in_ch=3,
            cond_dim=int(checkpoint["cond_dim"]),
            base_ch=int(config.get("base_channels", checkpoint.get("base_ch", 64))),
            time_dim=int(config.get("time_embedding_dim", checkpoint.get("time_dim", 128))),
        )
        model.load_state_dict(checkpoint["ema_model_state"], strict=True)
        models[family] = model.to(device).eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    schedule = x3_model.make_linear_schedule(
        int(checkpoints["prompted"].get("T", 1000)), device=device
    )
    save_steps = np.linspace(
        0, args.ddim_steps - 1, args.trajectory_snapshots, dtype=np.int64
    ).tolist()
    timestep_grid = np.linspace(
        schedule.T - 1, 0, args.ddim_steps, dtype=np.int64
    )
    trajectory_t = np.asarray([int(timestep_grid[index]) for index in save_steps])
    vocab = checkpoints["prompted"]["vocab"]
    width = max(2, len(str(args.num_queries - 1)))
    manifest_records = []

    for record in tqdm(records, desc="Sampling x3 queries", unit="query"):
        condition = torch.zeros((1, len(vocabulary)), device=device)
        for label in record["labels"]:
            condition[0, vocab[label]] = 1.0
        trajectory = x3_model.ddim_sample(
            model=models[str(record["family"])],
            sched=schedule,
            cond=condition,
            shape=(1, 3, 3, 3),
            seed=int(record["initial_seed"]),
            steps=args.ddim_steps,
            eta=0.0,
            device=str(device),
            save_steps=save_steps,
        )
        query_dir = args.output_dir / f"q{int(record['query_id']):0{width}d}"
        query_dir.mkdir(parents=True, exist_ok=True)
        trajectory_array = np.stack(
            [state.detach().cpu().numpy() for state in trajectory], axis=0
        )
        np.save(query_dir / "trajectory_xt.npy", trajectory_array)
        np.save(query_dir / "trajectory_t.npy", trajectory_t)
        np.save(query_dir / "final_state.npy", trajectory_array[-1])
        _save_preview(trajectory_array[-1, 0], query_dir / "final.png")
        saved_record = {
            **record,
            "dir": str(query_dir),
            "model_seed": model_seed,
            "sampler": "ddim_eta0",
        }
        manifest_records.append(saved_record)
        (query_dir / "query.json").write_text(
            json.dumps(saved_record, indent=2) + "\n", encoding="utf-8"
        )

    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest_records, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(records)} queries to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
