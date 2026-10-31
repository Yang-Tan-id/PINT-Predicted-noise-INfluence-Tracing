from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from pint.attribution.common import (
    attribution_metadata,
    datapoint_shard,
    load_query_parts,
    save_shard,
)
from pint.attribution.torch_adamw import adamw_full_direction_batch
from pint.query_gradients.common import x3_tracin_das_noise_seed
from pint.query_gradients.torch_projector import CountSketchProjector
from pint.query_gradients.x3 import _build_model, _checkpoints
from pint.training.backends.x3_dataset import ColorGridDataset


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute datapoint-sharded x3 PINT attribution")
    parser.add_argument("--experiment", choices=("x3",), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--query-gradient-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", default="all")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-adamw", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args(argv)
    if args.scope != "all":
        parser.error("x3 attribution currently uses scope 'all'")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    try:
        import torch
        import torch.nn.functional as functional
        from torch.func import functional_call, grad, vmap
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise SystemExit("x3 attribution requires train-x3 dependencies") from exc
    if args.device == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("--device gpu requested, but CUDA is unavailable")
    device = torch.device("cuda:0" if args.device != "cpu" and torch.cuda.is_available() else "cpu")

    query_manifest, query_parts = load_query_parts(args.query_gradient_dir)
    checkpoints = _checkpoints(args.model_dir)
    if len(query_parts) != len(checkpoints) - 1:
        raise RuntimeError("query gradients and x3 checkpoints have different pair counts")
    first_model, first_payload = _build_model(checkpoints[0], device)
    model_seed = int(first_payload["seed"])
    dataset_path = Path(
        first_payload.get("config", {}).get(
            "dataset", Path("data/x3/67_50000.csv")
        )
    )
    training_config = args.model_dir / "base" / "prompted" / "training_config.json"
    if training_config.is_file():
        configured = json.loads(training_config.read_text(encoding="utf-8"))
        dataset_path = Path(configured["dataset"])
    dataset = ColorGridDataset(str(dataset_path), grid_size=3)
    scope_indices = np.arange(len(dataset), dtype=np.int64)
    owned = datapoint_shard(len(scope_indices), args.shard_index, args.shard_count)
    train_indices = scope_indices[list(owned)]
    if not len(train_indices):
        raise RuntimeError("more GPU shards than x3 attribution datapoints")
    query_ids = np.asarray(query_parts[0]["query_ids"], dtype=np.int32)
    raw_scores = np.zeros((len(query_ids), len(train_indices)), dtype=np.float64)
    adamw_scores = (
        None if args.no_adamw
        else np.zeros((len(query_ids), len(train_indices)), dtype=np.float64)
    )

    from pint.training.backends.x3_model import make_linear_schedule

    schedule = make_linear_schedule(int(first_payload.get("T", 1000)), device=device)
    total_units = len(query_parts) * math.ceil(len(train_indices) / args.batch_size) * len(query_manifest["timesteps"])
    progress = tqdm(
        total=total_units,
        desc=f"x3 attribution shard {args.shard_index + 1}/{args.shard_count}",
        unit="batch-timestep",
        dynamic_ncols=True,
    )
    for checkpoint_index, part in enumerate(query_parts):
        model = first_model if checkpoint_index == 0 else _build_model(checkpoints[checkpoint_index], device)[0]
        payload = first_payload if checkpoint_index == 0 else torch.load(checkpoints[checkpoint_index], map_location="cpu")
        if int(payload["seed"]) != model_seed:
            raise RuntimeError("x3 checkpoint directory contains multiple model seeds")
        active = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = None
        if not args.no_adamw:
            if "optimizer_state" not in payload:
                raise RuntimeError(f"optimizer state is missing from {checkpoints[checkpoint_index]}")
            config = payload.get("config", {})
            optimizer = torch.optim.AdamW(
                active,
                lr=float(payload["learning_rate_at_checkpoint"]),
                betas=(float(config.get("adam_b1", 0.9)), float(config.get("adam_b2", 0.999))),
                eps=float(config.get("adam_eps", 1e-8)),
                weight_decay=float(config.get("weight_decay", 1e-4)),
            )
            optimizer.load_state_dict(payload["optimizer_state"])
        named_parameters = dict(model.named_parameters())
        projector = CountSketchProjector(
            active,
            int(query_manifest["projection_dim"]),
            model_seed,
            "tracin_das_parameter_projection",
            checkpoint_index,
        )
        timesteps = np.asarray(part["timesteps"], dtype=np.int32)
        features = np.asarray(part["query_features"], dtype=np.float32)
        if features.shape[:2] != (len(query_ids), len(timesteps)):
            raise ValueError("invalid x3 query-gradient feature shape")
        query_features = torch.from_numpy(features).to(device)
        learning_rate = float(payload["learning_rate_at_checkpoint"])

        def one_loss(params, clean, condition, timestep, noise):
            t = timestep.reshape(1).to(dtype=torch.long)
            noisy = (
                schedule.sqrt_alpha_bars[t].reshape(1, 1, 1) * clean
                + schedule.sqrt_one_minus_alpha_bars[t].reshape(1, 1, 1) * noise
            )
            prediction = functional_call(
                model, params, (noisy.unsqueeze(0), t, condition.unsqueeze(0))
            )[0]
            return functional.mse_loss(prediction, noise, reduction="mean")

        per_example_gradient = vmap(
            grad(one_loss), in_dims=(None, 0, 0, 0, 0), randomness="error"
        )
        for destination_start in range(0, len(train_indices), args.batch_size):
            destination_end = min(destination_start + args.batch_size, len(train_indices))
            batch_indices = train_indices[destination_start:destination_end]
            examples = [dataset[int(index)] for index in batch_indices]
            clean = torch.stack([item[0] for item in examples]).to(device)
            condition = torch.stack([item[1] for item in examples]).to(device)
            raw_contribution = torch.zeros(
                (len(query_ids), len(batch_indices)), device=device, dtype=torch.float32
            )
            adamw_contribution = torch.zeros_like(raw_contribution) if optimizer is not None else None
            for timestep_slot, timestep in enumerate(timesteps):
                generator = torch.Generator(device=device)
                generator.manual_seed(
                    x3_tracin_das_noise_seed(
                        checkpoint_index,
                        len(timesteps) - 1 - timestep_slot,
                        int(timestep),
                    )
                )
                shared_noise = torch.randn(
                    (1, *clean.shape[1:]), generator=generator, device=device, dtype=clean.dtype
                )
                noise = shared_noise.expand_as(clean)
                t = torch.full((len(clean),), int(timestep), device=device, dtype=torch.long)
                gradients = per_example_gradient(named_parameters, clean, condition, t, noise)
                gradient_batch = list(gradients.values())
                projected = projector.project_batch(gradient_batch)
                raw_contribution.add_(query_features[:, timestep_slot] @ projected.T)
                if optimizer is not None:
                    directions, reconstructed_lr = adamw_full_direction_batch(
                        active,
                        gradient_batch,
                        optimizer,
                        max_grad_norm=float(payload.get("config", {}).get("grad_clip_norm", 1.0)),
                    )
                    if not np.isclose(learning_rate, reconstructed_lr):
                        raise RuntimeError("optimizer LR differs from checkpoint LR")
                    assert adamw_contribution is not None
                    adamw_projected = projector.project_batch(directions)
                    adamw_contribution.add_(query_features[:, timestep_slot] @ adamw_projected.T)
                progress.update(1)
            raw_scores[:, destination_start:destination_end] += (
                raw_contribution.detach().double().cpu().numpy()
                * learning_rate
                / len(timesteps)
            )
            if adamw_scores is not None:
                assert adamw_contribution is not None
                adamw_scores[:, destination_start:destination_end] += (
                    adamw_contribution.detach().double().cpu().numpy()
                    * learning_rate
                    / len(timesteps)
                )
    progress.close()
    metadata = attribution_metadata(
        experiment="x3",
        query_manifest=args.query_manifest,
        query_gradient_manifest=query_manifest,
        scope=args.scope,
        scope_size=len(scope_indices),
        model_seed=model_seed,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        batch_size=args.batch_size,
        adamw_enabled=not args.no_adamw,
    )
    save_shard(
        args.output_dir,
        args.shard_index,
        raw_scores=raw_scores,
        adamw_scores=adamw_scores,
        train_indices=train_indices,
        query_ids=query_ids,
        metadata=metadata,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
