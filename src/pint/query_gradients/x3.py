from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np

from pint.query_gradients.common import (
    atomic_savez,
    batched,
    checkpoint_pair_range,
    part_path,
    read_query_manifest,
    stable_torch_seed,
    timestep_grid,
    validate_sampled_timesteps,
    write_run_manifest,
)
from pint.query_gradients.torch_projector import CountSketchProjector


CHECKPOINT_PATTERN = re.compile(r"epoch_(?P<epoch>\d+)\.pt$")


def _checkpoints(model_dir: Path) -> list[Path]:
    paths = []
    for path in (model_dir / "base" / "prompted").glob("epoch_*.pt"):
        match = CHECKPOINT_PATTERN.match(path.name)
        if match:
            paths.append((int(match.group("epoch")), path))
    if len(paths) < 2:
        raise FileNotFoundError(
            f"Need at least two prompted x3 checkpoints under {model_dir}"
        )
    return [path for _, path in sorted(paths)]


def _build_model(checkpoint_path: Path, device):
    import torch

    from pint.training.backends.x3_model import CondEpsModel

    payload = torch.load(checkpoint_path, map_location="cpu")
    config = payload.get("config", {})
    model = CondEpsModel(
        in_ch=3,
        cond_dim=int(payload["cond_dim"]),
        base_ch=int(config.get("base_channels", payload.get("base_ch", 64))),
        time_dim=int(config.get("time_embedding_dim", payload.get("time_dim", 128))),
    )
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    return model, payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute checkpoint-major x3 PINT gradients")
    parser.add_argument("--experiment", choices=("x3",), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--projection-dim", type=int, default=4096)
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--shard-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--shard-count", type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("x3 query gradients require the train-x3 dependencies") from exc
    if args.batch_size <= 0 or args.projection_dim <= 0:
        parser.error("--batch-size and --projection-dim must be positive")
    if args.device == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("--device gpu requested, but CUDA is unavailable")
    device = torch.device(
        "cuda:0" if args.device != "cpu" and torch.cuda.is_available() else "cpu"
    )
    checkpoints = _checkpoints(args.model_dir)
    try:
        pair_indices = checkpoint_pair_range(
            len(checkpoints) - 1, args.shard_index, args.shard_count
        )
    except ValueError as exc:
        parser.error(str(exc))
    if not pair_indices:
        return 0
    _, queries = read_query_manifest(args.query_manifest)
    current_model, current_payload = _build_model(checkpoints[pair_indices.start], device)
    timesteps = timestep_grid("x3", int(current_payload.get("T", 1000)))
    model_seed = int(current_payload["seed"])
    if any(int(query.get("model_seed", -1)) != model_seed for query in queries):
        raise RuntimeError(
            "x3 query bank and training checkpoints must use the same model seed"
        )
    write_run_manifest(
        args.output_dir,
        experiment="x3",
        query_manifest=args.query_manifest,
        checkpoints=checkpoints,
        timesteps=timesteps,
        projection_dim=args.projection_dim,
        batch_size=args.batch_size,
    )
    schedule = __import__(
        "pint.training.backends.x3_model", fromlist=["make_linear_schedule"]
    ).make_linear_schedule(int(current_payload.get("T", 1000)), device=device)
    vocab = current_payload["vocab"]

    endpoints = []
    conditions = []
    query_ids = []
    for query in queries:
        query_dir = Path(query["dir"])
        validate_sampled_timesteps(
            query_dir, timesteps, query_id=int(query["query_id"])
        )
        endpoint = np.load(query_dir / "final_state.npy").astype(np.float32)
        if endpoint.shape != (1, 3, 3, 3):
            raise ValueError(f"Unexpected x3 endpoint shape {endpoint.shape}: {query_dir}")
        condition = np.zeros((len(vocab),), dtype=np.float32)
        for label in query["labels"]:
            condition[int(vocab[label])] = 1.0
        endpoints.append(endpoint[0])
        conditions.append(condition)
        query_ids.append(int(query["query_id"]))

    for checkpoint_index in pair_indices:
        target_model, target_payload = _build_model(
            checkpoints[checkpoint_index + 1], device
        )
        if int(target_payload["seed"]) != model_seed:
            raise RuntimeError("x3 checkpoint directory contains multiple model seeds")
        output_path = part_path(args.output_dir, checkpoint_index)
        if output_path.is_file():
            current_model = target_model
            current_payload = target_payload
            continue

        active = [parameter for parameter in current_model.parameters() if parameter.requires_grad]
        projector = CountSketchProjector(
            active,
            args.projection_dim,
            model_seed,
            "traj_tracin_projection",
            checkpoint_index,
        )
        generator = torch.Generator(device=device)
        noise_by_timestep = []
        for timestep in timesteps:
            generator.manual_seed(
                stable_torch_seed(
                    model_seed, checkpoint_index, int(timestep), "query_noise"
                )
            )
            noise_by_timestep.append(
                torch.randn((1, 3, 3, 3), generator=generator, device=device)
            )

        all_features = []
        all_delta_norms = []
        for batch_start in range(0, len(queries), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(queries))
            x0 = torch.from_numpy(np.stack(endpoints[batch_start:batch_end])).to(device)
            condition = torch.from_numpy(
                np.stack(conditions[batch_start:batch_end])
            ).to(device)
            batch_features = []
            batch_norms = []
            for timestep, shared_noise in zip(timesteps, noise_by_timestep):
                t = torch.full(
                    (x0.shape[0],), int(timestep), device=device, dtype=torch.long
                )
                noise = shared_noise.expand_as(x0)
                xt = (
                    schedule.sqrt_alpha_bars[t].view(-1, 1, 1, 1) * x0
                    + schedule.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1) * noise
                )
                prediction = current_model(xt, t, condition)
                with torch.no_grad():
                    target = target_model(xt, t, condition)
                    delta = target - prediction.detach()
                timestep_features = []
                timestep_norms = []
                for local_index in range(x0.shape[0]):
                    scalar = (prediction[local_index] * delta[local_index]).sum()
                    gradients = torch.autograd.grad(
                        scalar,
                        active,
                        retain_graph=local_index + 1 < x0.shape[0],
                        allow_unused=False,
                    )
                    timestep_features.append(projector(gradients).detach().cpu().numpy())
                    timestep_norms.append(delta[local_index].float().norm().item())
                batch_features.append(np.stack(timestep_features))
                batch_norms.append(np.asarray(timestep_norms, dtype=np.float32))
            all_features.append(np.stack(batch_features, axis=1))
            all_delta_norms.append(np.stack(batch_norms, axis=1))
            print(
                f"checkpoint {checkpoint_index + 1}/{len(checkpoints) - 1}: "
                f"queries {batch_start + 1}-{batch_end}/{len(queries)}",
                flush=True,
            )
        features = np.concatenate(all_features).astype(np.float32)
        atomic_savez(
            output_path,
            query_features=features,
            projected_gradient_l2=np.linalg.norm(features, axis=-1).astype(np.float32),
            predicted_noise_delta_l2=np.concatenate(all_delta_norms).astype(np.float32),
            query_ids=np.asarray(query_ids, dtype=np.int32),
            timesteps=timesteps,
            checkpoint_index=np.asarray(checkpoint_index, dtype=np.int32),
            current_checkpoint=np.asarray(str(checkpoints[checkpoint_index])),
            target_checkpoint=np.asarray(str(checkpoints[checkpoint_index + 1])),
            normalization=np.asarray("raw"),
            projection_dim=np.asarray(args.projection_dim, dtype=np.int32),
        )
        current_model = target_model
        current_payload = target_payload
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
