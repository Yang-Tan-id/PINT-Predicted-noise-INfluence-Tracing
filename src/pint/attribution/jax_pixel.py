from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np

from pint.attribution.common import (
    attribution_metadata,
    datapoint_shard,
    load_query_parts,
    save_shard,
)
from pint.attribution_scope import load_scope, resolve_indices
from pint.query_gradients.jax_pixel import _checkpoints, _projection_specs


def _train_batch_function(model, projection_dim: int, tx, include_adamw: bool):
    import jax
    import jax.numpy as jnp

    scale = jnp.asarray(1.0 / np.sqrt(float(projection_dim)), dtype=jnp.float32)

    def project(gradients, specs):
        leaves, _ = jax.tree_util.tree_flatten(gradients)
        output = jnp.zeros((projection_dim,), dtype=jnp.float32)
        for gradient, (buckets, signs) in zip(leaves, specs):
            output = output.at[buckets].add(signs * jnp.ravel(gradient).astype(jnp.float32))
        output = output * scale
        return output / jnp.maximum(
            jnp.linalg.norm(output), jnp.asarray(1e-12, dtype=jnp.float32)
        )

    def loss(params, xt, timestep, condition, noise):
        t = jnp.full((xt.shape[0],), timestep, dtype=jnp.int32)
        prediction = model.apply({"params": params}, xt, t, condition, train=False)
        return jnp.mean((prediction.astype(jnp.float32) - noise.astype(jnp.float32)) ** 2)

    gradient = jax.grad(loss)

    def one(params, opt_state, xt, timestep, condition, noise, update_lr, specs):
        gradients = gradient(params, xt, timestep, condition, noise)
        raw = project(gradients, specs)
        if not include_adamw:
            return raw
        updates, _ = tx.update(gradients, opt_state, params)
        direction = jax.tree_util.tree_map(
            lambda update: update.astype(jnp.float32) / update_lr, updates
        )
        return raw, project(direction, specs)

    by_timestep = jax.vmap(
        one, in_axes=(None, None, 0, 0, None, 0, None, None)
    )
    by_example = jax.vmap(
        by_timestep, in_axes=(None, None, 0, None, 0, None, None, None)
    )
    return jax.jit(by_example)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute datapoint-sharded JAX PINT attribution")
    parser.add_argument("--experiment", choices=("shape3d", "celeba"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--query-gradient-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", default="paper_5k")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-adamw", action="store_true")
    parser.add_argument("--timestamp-chunk-size", type=int, default=10)
    parser.add_argument("--device", choices=("gpu", "cpu", "auto"), default="gpu")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.timestamp_chunk_size <= 0:
        parser.error("batch and timestamp chunk sizes must be positive")
    try:
        import jax
        import jax.numpy as jnp
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise SystemExit("Shape3D/CelebA attribution requires train-jax") from exc
    from pint.training.backends.jax_pixel import (
        CIFAR10Dataset,
        TrainConfig,
        _restore_checkpoint,
        build_model,
        choose_devices,
        create_train_state,
        learning_rate_at_step,
        make_diffusion_schedule,
    )

    query_manifest, query_parts = load_query_parts(args.query_gradient_dir)
    model_seed, checkpoints = _checkpoints(args.model_dir)
    if len(query_parts) != len(checkpoints) - 1:
        raise RuntimeError("query gradients and JAX checkpoints have different pair counts")
    with checkpoints[0].open("rb") as handle:
        first_payload = pickle.load(handle)
    config = TrainConfig(
        **{
            **first_payload["config"],
            "prefer_device": args.device,
            "use_data_parallel": False,
            "num_devices": 1,
        }
    )
    if int(config.seed) != model_seed:
        raise RuntimeError("JAX checkpoint filename and config model seeds differ")
    dataset = CIFAR10Dataset(
        root=config.data_root,
        batch_names=config.batch_names,
        use_test=config.use_test,
        class_names=config.class_names,
        normalize="minus_one_to_one",
        channels_last=True,
        cond_mode=config.cond_mode,
    )
    scope = load_scope(args.experiment, args.scope)
    scope_indices = resolve_indices(scope, len(dataset))
    owned = datapoint_shard(len(scope_indices), args.shard_index, args.shard_count)
    train_indices = scope_indices[list(owned)]
    if not len(train_indices):
        raise RuntimeError("more GPU shards than attribution datapoints")

    device = choose_devices(args.device)[0]
    model = build_model(config)
    steps_per_epoch = len(dataset) // int(config.batch_size)
    total_steps = steps_per_epoch * int(config.epochs)
    template = create_train_state(
        config,
        model,
        jax.random.PRNGKey(model_seed),
        device,
        total_steps,
    )
    schedule = make_diffusion_schedule(config.timesteps, config.beta_start, config.beta_end)
    alpha_bars = np.asarray(schedule.alphas_cumprod, dtype=np.float32)
    train_function = _train_batch_function(
        model,
        int(query_manifest["projection_dim"]),
        template.tx,
        not args.no_adamw,
    )
    query_ids = np.asarray(query_parts[0]["query_ids"], dtype=np.int32)
    raw_scores = np.zeros((len(query_ids), len(train_indices)), dtype=np.float64)
    adamw_scores = (
        None if args.no_adamw
        else np.zeros((len(query_ids), len(train_indices)), dtype=np.float64)
    )
    timestep_count = len(query_manifest["timesteps"])
    if timestep_count % args.timestamp_chunk_size:
        parser.error(
            "--timestamp-chunk-size must divide the fixed timestep count to "
            "preserve one compiled JAX kernel shape"
        )
    progress = tqdm(
        total=(
            len(query_parts)
            * math.ceil(len(train_indices) / args.batch_size)
            * (timestep_count // args.timestamp_chunk_size)
        ),
        desc=f"{args.experiment} attribution shard {args.shard_index + 1}/{args.shard_count}",
        unit="batch-chunk",
        dynamic_ncols=True,
    )
    for checkpoint_index, part in enumerate(query_parts):
        state, _epoch = _restore_checkpoint(str(checkpoints[checkpoint_index]), template)
        params = jax.device_put(state.params, device)
        opt_state = jax.device_put(state.opt_state, device)
        specs = _projection_specs(
            params,
            int(query_manifest["projection_dim"]),
            (model_seed, "traj_tracin_projection", checkpoint_index),
            device,
        )
        timesteps = np.asarray(part["timesteps"], dtype=np.int32)
        query_features = np.asarray(part["query_features"], dtype=np.float32)
        if query_features.shape[:2] != (len(query_ids), len(timesteps)):
            raise ValueError("invalid JAX query-gradient feature shape")
        query_features = query_features / np.maximum(
            np.linalg.norm(query_features, axis=-1, keepdims=True), 1e-12
        )
        shared_noise = []
        for timestep in timesteps:
            key = jax.random.PRNGKey(model_seed + 91_337)
            key = jax.random.fold_in(key, checkpoint_index)
            key = jax.random.fold_in(key, int(timestep))
            shared_noise.append(
                np.asarray(
                    jax.random.normal(
                        key, dataset.images[0:1].shape, dtype=jnp.float32
                    )
                )[0]
            )
        shared_noise = np.stack(shared_noise)
        # TrainState.step is the exact number of completed optimizer updates;
        # the checkpoint's last applied LR therefore belongs to step-1.
        completed_steps = int(np.asarray(jax.device_get(state.step)))
        step = max(0, min(total_steps - 1, completed_steps - 1))
        learning_rate = learning_rate_at_step(config, step, total_steps)
        next_step = max(0, min(total_steps - 1, completed_steps))
        update_learning_rate = learning_rate_at_step(config, next_step, total_steps)
        if not args.no_adamw and update_learning_rate <= 0:
            raise RuntimeError("cannot factor a zero learning rate from the AdamW update")
        for destination_start in range(0, len(train_indices), args.batch_size):
            real_count = min(args.batch_size, len(train_indices) - destination_start)
            batch_indices = train_indices[destination_start : destination_start + real_count]
            padded = list(map(int, batch_indices))
            while len(padded) < args.batch_size:
                padded.append(padded[-1])
            # Preserve the singleton model batch axis used by the original
            # per-datapoint loss before vmapping over datapoints/timesteps.
            clean = dataset.images[padded, None, ...]
            condition = dataset.labels[padded, None, ...]
            xt = np.stack(
                [
                    np.stack(
                        [
                            np.sqrt(float(alpha_bars[timestep])) * image
                            + np.sqrt(1.0 - float(alpha_bars[timestep])) * noise
                            for timestep, noise in zip(timesteps, shared_noise)
                        ]
                    )
                    for image in clean
                ]
            )
            raw_contribution = np.zeros((len(query_ids), real_count), dtype=np.float32)
            adamw_contribution = (
                None if args.no_adamw
                else np.zeros((len(query_ids), real_count), dtype=np.float32)
            )
            for chunk_start in range(0, len(timesteps), args.timestamp_chunk_size):
                chunk_end = chunk_start + args.timestamp_chunk_size
                projected = train_function(
                    params,
                    opt_state,
                    jax.device_put(jnp.asarray(xt[:, chunk_start:chunk_end]), device),
                    jax.device_put(jnp.asarray(timesteps[chunk_start:chunk_end]), device),
                    jax.device_put(jnp.asarray(condition), device),
                    jax.device_put(jnp.asarray(shared_noise[chunk_start:chunk_end]), device),
                    jnp.asarray(update_learning_rate, dtype=jnp.float32),
                    specs,
                )
                if args.no_adamw:
                    raw_projected = projected
                    adamw_projected = None
                else:
                    raw_projected, adamw_projected = projected
                raw_projected_np = np.asarray(jax.device_get(raw_projected))[:real_count]
                raw_contribution += np.einsum(
                    "qtd,btd->qb",
                    query_features[:, chunk_start:chunk_end],
                    raw_projected_np,
                    optimize=True,
                )
                if adamw_projected is not None:
                    assert adamw_contribution is not None
                    adamw_projected_np = np.asarray(jax.device_get(adamw_projected))[:real_count]
                    adamw_contribution += np.einsum(
                        "qtd,btd->qb",
                        query_features[:, chunk_start:chunk_end],
                        adamw_projected_np,
                        optimize=True,
                    )
                progress.update(1)
            raw_scores[:, destination_start : destination_start + real_count] += (
                raw_contribution.astype(np.float64) * learning_rate / len(timesteps)
            )
            if adamw_scores is not None:
                assert adamw_contribution is not None
                adamw_scores[:, destination_start : destination_start + real_count] += (
                    adamw_contribution.astype(np.float64) * learning_rate / len(timesteps)
                )
    progress.close()
    metadata = attribution_metadata(
        experiment=args.experiment,
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
