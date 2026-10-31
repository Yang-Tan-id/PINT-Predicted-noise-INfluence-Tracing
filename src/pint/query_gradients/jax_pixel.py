from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path
from typing import Sequence

import numpy as np

from pint.query_gradients.common import (
    atomic_savez,
    checkpoint_pair_range,
    part_path,
    read_query_manifest,
    stable_seed,
    timestep_grid,
    validate_sampled_timesteps,
    write_run_manifest,
)


CHECKPOINT_PATTERN = re.compile(r"seed_(?P<seed>-?\d+)_epoch_(?P<epoch>\d+)\.ckpt$")


def _checkpoints(model_dir: Path) -> tuple[int, list[Path]]:
    found = []
    for path in model_dir.glob("seed_*_epoch_*.ckpt"):
        match = CHECKPOINT_PATTERN.match(path.name)
        if match:
            found.append((int(match.group("seed")), int(match.group("epoch")), path))
    seeds = {seed for seed, _, _ in found}
    if len(seeds) != 1:
        raise RuntimeError(
            f"Expected one JAX model seed under {model_dir}, found {sorted(seeds)}"
        )
    if len(found) < 2:
        raise FileNotFoundError(f"Need at least two JAX checkpoints under {model_dir}")
    seed = seeds.pop()
    return seed, [path for _, _, path in sorted(found, key=lambda item: item[1])]


def _projection_specs(params, dimension: int, seed_parts: tuple[object, ...], device):
    import jax
    import jax.numpy as jnp

    leaves, _ = jax.tree_util.tree_flatten(params)
    specs = []
    for index, leaf in enumerate(leaves):
        count = int(np.prod(leaf.shape))
        rng = np.random.default_rng(
            stable_seed(*seed_parts, "tensor", index, count, dimension)
        )
        buckets = jax.device_put(
            jnp.asarray(rng.integers(0, dimension, count, dtype=np.int32)), device
        )
        signs = jax.device_put(
            jnp.asarray(
                rng.integers(0, 2, count, dtype=np.int8).astype(np.float32) * 2 - 1
            ),
            device,
        )
        specs.append((buckets, signs))
    return tuple(specs)


def _batch_function(model, projection_dim: int):
    import jax
    import jax.numpy as jnp

    scale = jnp.asarray(1.0 / np.sqrt(float(projection_dim)), dtype=jnp.float32)

    def project(gradients, specs):
        gradient_leaves, _ = jax.tree_util.tree_flatten(gradients)
        output = jnp.zeros((projection_dim,), dtype=jnp.float32)
        for gradient, (buckets, signs) in zip(gradient_leaves, specs):
            output = output.at[buckets].add(
                signs * jnp.ravel(gradient).astype(jnp.float32)
            )
        return output * scale

    def scalar(current_params, target_params, xt, timestep, condition):
        t = jnp.full((xt.shape[0],), timestep, dtype=jnp.int32)
        prediction = model.apply(
            {"params": current_params}, xt, t, condition, train=False
        )
        target = jax.lax.stop_gradient(
            model.apply({"params": target_params}, xt, t, condition, train=False)
        )
        delta = jax.lax.stop_gradient(target - prediction)
        norm = jnp.linalg.norm(delta.astype(jnp.float32))
        direction = delta / jnp.maximum(norm, jnp.asarray(1e-12, dtype=jnp.float32))
        return jnp.mean(prediction * direction), norm

    value_and_grad = jax.value_and_grad(scalar, has_aux=True)

    def one(current_params, target_params, xt, timestep, condition, specs):
        (_value, delta_norm), gradient = value_and_grad(
            current_params, target_params, xt, timestep, condition
        )
        return project(gradient, specs), delta_norm

    by_timestep = jax.vmap(one, in_axes=(None, None, 0, 0, None, None))
    by_query = jax.vmap(
        by_timestep, in_axes=(None, None, 0, None, 0, None)
    )
    return jax.jit(by_query)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute checkpoint-major JAX PINT gradients")
    parser.add_argument("--experiment", choices=("shape3d", "celeba"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--projection-dim", type=int, default=4096)
    parser.add_argument("--timestamp-chunk-size", type=int, default=10)
    parser.add_argument("--device", choices=("gpu", "cpu", "auto"), default="gpu")
    parser.add_argument("--shard-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--shard-count", type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.projection_dim <= 0 or args.timestamp_chunk_size <= 0:
        parser.error("batch, projection, and timestamp chunk sizes must be positive")

    try:
        import jax
        import jax.numpy as jnp
        from flax.serialization import from_bytes
    except ImportError as exc:
        raise SystemExit("Shape3D/CelebA query gradients require train-jax") from exc
    from pint.training.backends.jax_pixel import (
        TrainConfig,
        _restore_checkpoint,
        build_model,
        choose_devices,
        create_train_state,
        make_diffusion_schedule,
    )

    model_seed, checkpoints = _checkpoints(args.model_dir)
    try:
        pair_indices = checkpoint_pair_range(
            len(checkpoints) - 1, args.shard_index, args.shard_count
        )
    except ValueError as exc:
        parser.error(str(exc))
    if not pair_indices:
        return 0
    manifest_metadata, queries = read_query_manifest(args.query_manifest)
    with checkpoints[pair_indices.start].open("rb") as handle:
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
    if "model_seed" in manifest_metadata and int(manifest_metadata["model_seed"]) != model_seed:
        raise RuntimeError("Query bank and checkpoint model seeds differ")
    if any(int(query.get("model_seed", -1)) != model_seed for query in queries):
        raise RuntimeError("Query bank and checkpoint model seeds differ")
    device = choose_devices(args.device)[0]
    model = build_model(config)
    template = create_train_state(config, model, jax.random.PRNGKey(model_seed), device)
    current_state = from_bytes(template, first_payload["state_bytes"])
    del first_payload
    current_params = jax.device_put(current_state.params, device)
    schedule = make_diffusion_schedule(config.timesteps, config.beta_start, config.beta_end)
    alpha_bars = np.asarray(schedule.alphas_cumprod, dtype=np.float32)
    timesteps = timestep_grid(args.experiment, config.timesteps)
    write_run_manifest(
        args.output_dir,
        experiment=args.experiment,
        query_manifest=args.query_manifest,
        checkpoints=checkpoints,
        timesteps=timesteps,
        projection_dim=args.projection_dim,
        batch_size=args.batch_size,
    )

    endpoints = []
    conditions = []
    query_ids = []
    label_names = manifest_metadata.get("label_names", [])
    label_to_index = {str(label): index for index, label in enumerate(label_names)}
    for query in queries:
        query_dir = Path(query["dir"])
        validate_sampled_timesteps(
            query_dir, timesteps, query_id=int(query["query_id"])
        )
        endpoint = np.load(query_dir / "final_state.npy").astype(np.float32)
        if endpoint.shape != (1, config.image_size, config.image_size, config.in_channels):
            raise ValueError(f"Unexpected endpoint shape {endpoint.shape} for query {query['query_id']}")
        if args.experiment == "celeba":
            condition = np.asarray(query["labels"], dtype=np.float32)[None, :]
        else:
            condition = np.zeros((1, config.num_classes), dtype=np.float32)
            for label in query["labels"]:
                condition[0, label_to_index[str(label)]] = 1.0
        endpoints.append(endpoint)
        conditions.append(condition)
        query_ids.append(int(query["query_id"]))

    # Projection arrays are dynamic arguments, so one compiled executable is
    # reused across checkpoints even though every checkpoint has its own seed.
    batch_function = _batch_function(model, args.projection_dim)
    for checkpoint_index in pair_indices:
        target_state, _ = _restore_checkpoint(
            str(checkpoints[checkpoint_index + 1]), template
        )
        target_params = jax.device_put(target_state.params, device)
        output_path = part_path(args.output_dir, checkpoint_index)
        if output_path.is_file():
            current_params = target_params
            continue
        projection_specs = _projection_specs(
            current_params,
            args.projection_dim,
            (model_seed, "traj_tracin_projection", checkpoint_index),
            device,
        )
        shared_noise = []
        for timestep in timesteps:
            key = jax.random.PRNGKey(model_seed + 91_337)
            key = jax.random.fold_in(key, checkpoint_index)
            key = jax.random.fold_in(key, int(timestep))
            shared_noise.append(
                np.asarray(
                    jax.random.normal(key, endpoints[0].shape, dtype=jnp.float32)
                )
            )
        shared_noise = np.stack(shared_noise)
        checkpoint_features = []
        checkpoint_gradient_norms = []
        checkpoint_norms = []
        for start in range(0, len(queries), args.batch_size):
            real_count = min(args.batch_size, len(queries) - start)
            batch_endpoints = list(endpoints[start : start + real_count])
            batch_conditions = list(conditions[start : start + real_count])
            while len(batch_endpoints) < args.batch_size:
                batch_endpoints.append(batch_endpoints[-1])
                batch_conditions.append(batch_conditions[-1])
            x0 = np.stack(batch_endpoints)
            xt = np.stack(
                [
                    np.stack(
                        [
                            np.sqrt(float(alpha_bars[timestep])) * endpoint
                            + np.sqrt(1.0 - float(alpha_bars[timestep])) * noise
                            for timestep, noise in zip(timesteps, shared_noise)
                        ]
                    )
                    for endpoint in x0
                ]
            )
            feature_chunks = []
            norm_chunks = []
            for chunk_start in range(0, len(timesteps), args.timestamp_chunk_size):
                chunk_end = min(chunk_start + args.timestamp_chunk_size, len(timesteps))
                features, norms = batch_function(
                    current_params,
                    target_params,
                    jax.device_put(jnp.asarray(xt[:, chunk_start:chunk_end]), device),
                    jax.device_put(jnp.asarray(timesteps[chunk_start:chunk_end]), device),
                    jax.device_put(jnp.asarray(np.stack(batch_conditions)), device),
                    projection_specs,
                )
                feature_chunks.append(np.asarray(jax.device_get(features)))
                norm_chunks.append(np.asarray(jax.device_get(norms)))
            features = np.concatenate(feature_chunks, axis=1)[:real_count]
            checkpoint_gradient_norms.append(
                np.linalg.norm(features, axis=-1).astype(np.float32)
            )
            checkpoint_features.append(features.astype(np.float32))
            checkpoint_norms.append(
                np.concatenate(norm_chunks, axis=1)[:real_count].astype(np.float32)
            )
            print(
                f"checkpoint {checkpoint_index + 1}/{len(checkpoints) - 1}: "
                f"queries {start + 1}-{start + real_count}/{len(queries)}",
                flush=True,
            )
        atomic_savez(
            output_path,
            query_features=np.concatenate(checkpoint_features),
            projected_gradient_l2=np.concatenate(checkpoint_gradient_norms),
            predicted_noise_delta_l2=np.concatenate(checkpoint_norms),
            query_ids=np.asarray(query_ids, dtype=np.int32),
            timesteps=timesteps,
            checkpoint_index=np.asarray(checkpoint_index, dtype=np.int32),
            current_checkpoint=np.asarray(str(checkpoints[checkpoint_index])),
            target_checkpoint=np.asarray(str(checkpoints[checkpoint_index + 1])),
            normalization=np.asarray("predicted_noise_delta_l2"),
            projection_dim=np.asarray(args.projection_dim, dtype=np.int32),
        )
        current_params = target_params
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
