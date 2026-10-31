from __future__ import annotations

import argparse
import math
import pickle
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
from pint.query_gradients.common import checkpoint_pair_range, read_query_manifest, timestep_grid
from pint.query_gradients.jax_pixel import _checkpoints, _projection_specs


def _feature_function(model, projection_dim: int, normalize_each: bool):
    import jax
    import jax.numpy as jnp

    scale = jnp.asarray(1.0 / np.sqrt(float(projection_dim)), dtype=jnp.float32)

    def project(gradient, specs):
        leaves, _ = jax.tree_util.tree_flatten(gradient)
        output = jnp.zeros((projection_dim,), dtype=jnp.float32)
        for leaf, (buckets, signs) in zip(leaves, specs):
            output = output.at[buckets].add(signs * jnp.ravel(leaf).astype(jnp.float32))
        return output * scale

    def one_loss(params, endpoint, condition, timestep, noise):
        t = jnp.full((endpoint.shape[0],), timestep, dtype=jnp.int32)
        # The caller supplies x_t so the exact training schedule remains outside
        # the compiled model-only gradient function.
        prediction = model.apply({"params": params}, endpoint, t, condition, train=False)
        return jnp.mean(jnp.square(prediction.astype(jnp.float32) - noise.astype(jnp.float32)))

    gradient = jax.grad(one_loss)

    def one(params, xt, condition, timesteps, noises, specs):
        def per_timestep(current_xt, timestep, noise):
            value = gradient(params, current_xt, condition, timestep, noise)
            if normalize_each:
                norm_sq = sum(
                    jnp.vdot(leaf.astype(jnp.float32), leaf.astype(jnp.float32)).real
                    for leaf in jax.tree_util.tree_leaves(value)
                )
                denominator = jnp.sqrt(jnp.maximum(norm_sq, 1e-16))
                value = jax.tree_util.tree_map(lambda leaf: leaf / denominator, value)
            return project(value, specs)

        features = jax.vmap(per_timestep)(xt, timesteps, noises)
        return jnp.mean(features, axis=0)

    return jax.jit(jax.vmap(one, in_axes=(None, 0, 0, None, 0, None)))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("shape3d", "celeba"), required=True)
    parser.add_argument("--method", choices=("gas", "tracincp", "retrac"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--projection-dim", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("gpu", "cpu", "auto"), default="gpu")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        import jax
        import jax.numpy as jnp
        from flax.serialization import from_bytes
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise SystemExit("Shape3D/CelebA baselines require train-jax") from exc
    from pint.training.backends.jax_pixel import (
        TrainConfig,
        build_model,
        choose_devices,
        create_train_state,
        make_diffusion_schedule,
    )

    spec = baseline_spec(args.method, args.experiment)
    tag = run_tag(spec, args.projection_dim)
    model_seed, checkpoints = _checkpoints(args.model_dir)
    source_count = len(checkpoints) - 1 if args.method == "retrac" else len(checkpoints)
    owned = checkpoint_pair_range(source_count, args.shard_index, args.shard_count)
    if not owned:
        return 0
    manifest_metadata, queries = read_query_manifest(args.query_manifest)
    with checkpoints[owned.start].open("rb") as handle:
        first_payload = pickle.load(handle)
    config = TrainConfig(
        **{
            **first_payload["config"],
            "prefer_device": args.device,
            "use_data_parallel": False,
            "num_devices": 1,
        }
    )
    device = choose_devices(args.device)[0]
    model = build_model(config)
    template = create_train_state(config, model, jax.random.PRNGKey(model_seed), device)
    schedule = make_diffusion_schedule(config.timesteps, config.beta_start, config.beta_end)
    alpha_bars = np.asarray(schedule.alphas_cumprod, dtype=np.float32)
    timesteps = timestep_grid(args.experiment, config.timesteps)
    write_query_manifest(
        args.output_dir,
        spec=spec,
        tag=tag,
        projection_dim=args.projection_dim,
        query_manifest=args.query_manifest,
        checkpoints=checkpoints,
        source_checkpoint_count=source_count,
    )
    label_names = manifest_metadata.get("label_names", [])
    label_to_index = {str(label): index for index, label in enumerate(label_names)}
    endpoints, conditions, query_ids = [], [], []
    for query in queries:
        endpoint = np.load(Path(query["dir"]) / "final_state.npy").astype(np.float32)
        if args.experiment == "celeba":
            condition = np.asarray(query["labels"], dtype=np.float32)[None, :]
        else:
            condition = np.zeros((1, config.num_classes), dtype=np.float32)
            for label in query["labels"]:
                condition[0, label_to_index[str(label)]] = 1.0
        endpoints.append(endpoint)
        conditions.append(condition)
        query_ids.append(int(query["query_id"]))
    feature_function = _feature_function(model, args.projection_dim, args.method == "retrac")
    progress = tqdm(
        total=len(owned) * math.ceil(len(queries) / args.batch_size),
        desc=f"{args.experiment} {args.method} query shard {args.shard_index + 1}/{args.shard_count}",
        unit="batch",
        dynamic_ncols=True,
    )
    for checkpoint_index in owned:
        output_path = query_part_path(args.output_dir, checkpoint_index, tag)
        if output_path.is_file():
            progress.update(math.ceil(len(queries) / args.batch_size))
            continue
        with checkpoints[checkpoint_index].open("rb") as handle:
            payload = pickle.load(handle)
        state = from_bytes(template, payload["state_bytes"])
        params = jax.device_put(state.params, device)
        specs = _projection_specs(
            params,
            args.projection_dim,
            (model_seed, "traj_tracin_projection", checkpoint_index),
            device,
        )
        all_features = []
        for start in range(0, len(queries), args.batch_size):
            end = min(start + args.batch_size, len(queries))
            real_count = end - start
            xt_batch, noise_batch = [], []
            for local_index, endpoint in enumerate(endpoints[start:end], start=start):
                query_id = query_ids[local_index]
                query_xt, query_noise = [], []
                key = jax.random.PRNGKey(model_seed + 487_301)
                for value in (query_id, checkpoint_index, 100_001):
                    key = jax.random.fold_in(key, value)
                noise_bank = np.asarray(
                    jax.random.normal(
                        key, (len(timesteps),) + endpoint.shape, dtype=jnp.float32
                    )
                )
                for timestep, noise in zip(timesteps, noise_bank):
                    noisy = (
                        np.sqrt(float(alpha_bars[timestep])) * endpoint
                        + np.sqrt(1.0 - float(alpha_bars[timestep])) * noise
                    )
                    query_xt.append(noisy)
                    query_noise.append(noise)
                xt_batch.append(np.stack(query_xt))
                noise_batch.append(np.stack(query_noise))
            condition_batch = list(conditions[start:end])
            while len(xt_batch) < args.batch_size:
                xt_batch.append(xt_batch[-1])
                noise_batch.append(noise_batch[-1])
                condition_batch.append(condition_batch[-1])
            features = feature_function(
                params,
                jax.device_put(jnp.asarray(np.stack(xt_batch)), device),
                jax.device_put(jnp.asarray(np.stack(condition_batch)), device),
                jax.device_put(jnp.asarray(timesteps), device),
                jax.device_put(jnp.asarray(np.stack(noise_batch)), device),
                specs,
            )
            all_features.append(
                np.asarray(jax.device_get(features), dtype=np.float32)[:real_count]
            )
            progress.update(1)
        atomic_savez(
            output_path,
            query_features=np.concatenate(all_features),
            query_ids=np.asarray(query_ids, dtype=np.int32),
            timesteps=timesteps,
            checkpoint_index=np.asarray(checkpoint_index, dtype=np.int32),
            checkpoint=np.asarray(str(checkpoints[checkpoint_index])),
            tag=np.asarray(tag),
            method=np.asarray(args.method),
            query_reduction=np.asarray(spec.query_reduction),
            projection_dim=np.asarray(args.projection_dim, dtype=np.int32),
        )
    progress.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
