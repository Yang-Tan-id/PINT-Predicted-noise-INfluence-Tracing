from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from pint.baselines.common import (
    atomic_savez,
    baseline_spec,
    query_part_path,
    run_tag,
    write_query_manifest,
)
from pint.query_gradients.common import (
    checkpoint_pair_range,
    read_query_manifest,
    stable_torch_seed,
)
from pint.query_gradients.torch_projector import CountSketchProjector
from pint.query_gradients.x3 import _build_model, _checkpoints


def _timesteps(method: str, total: int) -> np.ndarray:
    if method == "retrac":
        return np.asarray(
            [int(round(1 + position * (total - 2) / 99.0)) for position in range(100)],
            dtype=np.int32,
        )
    return np.asarray([int(index * (total - 1) / 99) for index in range(100)], dtype=np.int32)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("x3",), required=True)
    parser.add_argument("--method", choices=("gas", "tracincp", "retrac"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--projection-dim", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        import torch
        import torch.nn.functional as functional
        from torch.func import functional_call, grad, vmap
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise SystemExit("x3 baselines require the train-x3 dependencies") from exc
    if args.device == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("--device gpu requested, but CUDA is unavailable")
    device = torch.device("cuda:0" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    spec = baseline_spec(args.method, "x3")
    tag = run_tag(spec, args.projection_dim)
    checkpoints = _checkpoints(args.model_dir)
    # ReTrac and the established x3 endpoint run both use source checkpoints
    # that have a following interval.
    owned = checkpoint_pair_range(len(checkpoints) - 1, args.shard_index, args.shard_count)
    if not owned:
        return 0
    _, queries = read_query_manifest(args.query_manifest)
    first_model, first_payload = _build_model(checkpoints[owned.start], device)
    model_seed = int(first_payload["seed"])
    timesteps = _timesteps(args.method, int(first_payload.get("T", 1000)))
    write_query_manifest(
        args.output_dir,
        spec=spec,
        tag=tag,
        projection_dim=args.projection_dim,
        query_manifest=args.query_manifest,
        checkpoints=checkpoints,
        source_checkpoint_count=len(checkpoints) - 1,
    )
    vocab = first_payload["vocab"]
    endpoints, conditions, query_ids = [], [], []
    for query in queries:
        endpoint = np.load(Path(query["dir"]) / "final_state.npy").astype(np.float32)
        condition = np.zeros((len(vocab),), dtype=np.float32)
        for label in query["labels"]:
            condition[int(vocab[label])] = 1.0
        endpoints.append(endpoint[0])
        conditions.append(condition)
        query_ids.append(int(query["query_id"]))
    endpoints_tensor = torch.from_numpy(np.stack(endpoints)).to(device)
    conditions_tensor = torch.from_numpy(np.stack(conditions)).to(device)
    from pint.training.backends.x3_model import make_linear_schedule, q_sample

    schedule = make_linear_schedule(int(first_payload.get("T", 1000)), device=device)
    progress = tqdm(
        total=len(owned) * len(timesteps) * math.ceil(len(queries) / args.batch_size),
        desc=f"x3 {args.method} query shard {args.shard_index + 1}/{args.shard_count}",
        unit="batch-timestep",
        dynamic_ncols=True,
    )
    for checkpoint_index in owned:
        path = query_part_path(args.output_dir, checkpoint_index, tag)
        if path.is_file():
            progress.update(len(timesteps) * math.ceil(len(queries) / args.batch_size))
            continue
        model, payload = (
            (first_model, first_payload)
            if checkpoint_index == owned.start
            else _build_model(checkpoints[checkpoint_index], device)
        )
        if int(payload["seed"]) != model_seed:
            raise RuntimeError("x3 checkpoint directory contains multiple model seeds")
        named = dict(model.named_parameters())
        active = tuple(named.values())
        projection_namespace = (
            "diffusion_retrac_projection"
            if args.method == "retrac"
            else "endpoint_tracin_adamw_mc10_projection"
        )
        projector = CountSketchProjector(
            active, args.projection_dim, model_seed, projection_namespace, checkpoint_index
        )

        def loss_one(parameters, clean, condition, timestep, noise):
            t = timestep.reshape(1)
            noisy = q_sample(clean.unsqueeze(0), t, noise.unsqueeze(0), schedule)
            prediction = functional_call(model, parameters, (noisy, t, condition.unsqueeze(0)))
            return functional.mse_loss(prediction, noise.unsqueeze(0))

        gradient_one = vmap(grad(loss_one), in_dims=(None, 0, 0, 0, 0))
        by_timestamp: list[np.ndarray] = []
        retrac_total = torch.zeros(
            (len(queries), args.projection_dim), device=device, dtype=torch.float32
        )
        for timestamp_index, timestep in enumerate(timesteps):
            generator = torch.Generator(device=device)
            generator.manual_seed(
                stable_torch_seed(
                    model_seed,
                    "diffusion_retrac_query_noise" if args.method == "retrac" else "endpoint_tracin_query_mc10",
                    checkpoint_index,
                    int(timestep) if args.method == "retrac" else timestamp_index,
                )
            )
            noises = torch.randn(
                (len(queries), spec.mc_samples, *endpoints_tensor.shape[1:]),
                generator=generator,
                device=device,
                dtype=endpoints_tensor.dtype,
            )
            timestamp_features = []
            for start in range(0, len(queries), args.batch_size):
                end = min(start + args.batch_size, len(queries))
                count = end - start
                clean = endpoints_tensor[start:end, None].expand(
                    count, spec.mc_samples, *endpoints_tensor.shape[1:]
                ).reshape(-1, *endpoints_tensor.shape[1:])
                condition = conditions_tensor[start:end, None].expand(
                    count, spec.mc_samples, conditions_tensor.shape[-1]
                ).reshape(-1, conditions_tensor.shape[-1])
                t = torch.full(
                    (count * spec.mc_samples,), int(timestep), device=device, dtype=torch.long
                )
                gradients = gradient_one(
                    named,
                    clean,
                    condition,
                    t,
                    noises[start:end].reshape(-1, *endpoints_tensor.shape[1:]),
                )
                means = [
                    value.reshape(count, spec.mc_samples, *value.shape[1:]).mean(dim=1)
                    for value in gradients.values()
                ]
                if args.method == "retrac":
                    norm = torch.sqrt(
                        sum(value.float().flatten(1).square().sum(dim=1) for value in means)
                    ).clamp_min(1e-12)
                    means = [
                        value / norm.reshape((count,) + (1,) * (value.ndim - 1))
                        for value in means
                    ]
                projected = projector.project_batch(means)
                timestamp_features.append(projected)
                progress.update(1)
            matrix = torch.cat(timestamp_features)
            if args.method == "retrac":
                retrac_total.add_(matrix, alpha=1.0 / len(timesteps))
            else:
                by_timestamp.append(matrix.cpu().numpy())
        features = (
            retrac_total.cpu().numpy()
            if args.method == "retrac"
            else np.stack(by_timestamp, axis=1)
        ).astype(np.float32)
        atomic_savez(
            path,
            query_features=features,
            query_ids=np.asarray(query_ids, dtype=np.int32),
            timesteps=timesteps,
            checkpoint_index=np.asarray(checkpoint_index, dtype=np.int32),
            checkpoint=np.asarray(str(checkpoints[checkpoint_index])),
            tag=np.asarray(tag),
            method=np.asarray(args.method),
            query_reduction=np.asarray(spec.query_reduction),
            projection_dim=np.asarray(args.projection_dim, dtype=np.int32),
        )
        del model, payload, named, active, projector
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    progress.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
