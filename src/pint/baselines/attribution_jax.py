from __future__ import annotations

import argparse
import hashlib
import math
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np

from pint.attribution.common import datapoint_shard, save_shard
from pint.attribution_scope import load_scope, resolve_indices
from pint.baselines.common import (
    baseline_spec, load_query_run, orient_adamw_scores, run_tag,
    validate_query_source,
)
from pint.query_gradients.jax_pixel import _checkpoints, _projection_specs


def _feature_function(model, projection_dim: int, tx, include_adamw: bool, full_l2: bool):
    import jax
    import jax.numpy as jnp

    scale = jnp.asarray(1.0 / np.sqrt(float(projection_dim)), dtype=jnp.float32)

    def project(tree, specs, normalize_full):
        leaves, _ = jax.tree_util.tree_flatten(tree)
        denominator = jnp.asarray(1.0, dtype=jnp.float32)
        if normalize_full:
            denominator = jnp.sqrt(
                jnp.maximum(
                    sum(jnp.vdot(x.astype(jnp.float32), x.astype(jnp.float32)).real for x in leaves),
                    1e-16,
                )
            )
        output = jnp.zeros((projection_dim,), dtype=jnp.float32)
        for leaf, (buckets, signs) in zip(leaves, specs):
            output = output.at[buckets].add(signs * jnp.ravel(leaf).astype(jnp.float32) / denominator)
        return output * scale

    def endpoint_loss(params, clean, condition, timesteps, noises, alpha_bars):
        noisy = (
            jnp.sqrt(alpha_bars).reshape((-1,) + (1,) * clean.ndim) * clean[None, ...]
            + jnp.sqrt(1.0 - alpha_bars).reshape((-1,) + (1,) * clean.ndim) * noises
        )
        count = timesteps.shape[0]
        flat_noisy = noisy.reshape((count * clean.shape[0],) + clean.shape[1:])
        flat_noise = noises.reshape(flat_noisy.shape)
        flat_t = jnp.repeat(timesteps, clean.shape[0])
        flat_condition = jnp.broadcast_to(condition[None, ...], (count,) + condition.shape).reshape(
            (count * condition.shape[0],) + condition.shape[1:]
        )
        prediction = model.apply({"params": params}, flat_noisy, flat_t, flat_condition, train=False)
        return jnp.mean(jnp.square(prediction.astype(jnp.float32) - flat_noise.astype(jnp.float32)))

    gradient = jax.grad(endpoint_loss)

    def one(params, opt_state, clean, condition, timesteps, noises, alpha_bars, update_lr, specs):
        gradients = gradient(params, clean, condition, timesteps, noises, alpha_bars)
        raw = project(gradients, specs, full_l2)
        if not include_adamw:
            return raw
        updates, _ = tx.update(gradients, opt_state, params)
        direction = jax.tree_util.tree_map(lambda value: value.astype(jnp.float32) / update_lr, updates)
        return raw, project(direction, specs, full_l2)

    return jax.jit(jax.vmap(one, in_axes=(None, None, 0, 0, None, 0, None, None, None)))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("shape3d", "celeba"), required=True)
    parser.add_argument("--method", choices=("gas", "tracincp", "retrac"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--query-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--projection-dim", type=int, default=4096)
    parser.add_argument("--scope", default="paper_5k")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-adamw", action="store_true")
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
        CIFAR10Dataset,
        TrainConfig,
        build_model,
        choose_devices,
        create_train_state,
        learning_rate_at_step,
        make_diffusion_schedule,
    )

    spec = baseline_spec(args.method, args.experiment)
    tag = run_tag(spec, args.projection_dim)
    manifest, parts = load_query_run(args.query_dir, tag)
    validate_query_source(manifest, args.query_manifest)
    model_seed, checkpoints = _checkpoints(args.model_dir)
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
    destination = {int(index): slot for slot, index in enumerate(train_indices)}
    device = choose_devices(args.device)[0]
    model = build_model(config)
    steps_per_epoch = len(dataset) // int(config.batch_size)
    total_steps = steps_per_epoch * int(config.epochs)
    template = create_train_state(config, model, jax.random.PRNGKey(model_seed), device, total_steps)
    schedule = make_diffusion_schedule(config.timesteps, config.beta_start, config.beta_end)
    alpha_bars_all = np.asarray(schedule.alphas_cumprod, dtype=np.float32)
    feature_function = _feature_function(
        model, args.projection_dim, template.tx, not args.no_adamw, args.method == "retrac"
    )
    query_ids = np.asarray(parts[0]["query_ids"], dtype=np.int32)
    raw_scores = np.zeros((len(query_ids), len(train_indices)), dtype=np.float64)
    adamw_scores = None if args.no_adamw else np.zeros_like(raw_scores)
    progress = tqdm(total=len(parts) * math.ceil(len(train_indices) / args.batch_size), desc=f"{args.experiment} {args.method} attribution", unit="batch", dynamic_ncols=True)
    for checkpoint_index, part in enumerate(parts):
        with checkpoints[checkpoint_index].open("rb") as handle:
            payload = pickle.load(handle)
        state = from_bytes(template, payload["state_bytes"])
        params = jax.device_put(state.params, device)
        opt_state = jax.device_put(state.opt_state, device)
        specs = _projection_specs(params, args.projection_dim, (model_seed, "traj_tracin_projection", checkpoint_index), device)
        query_features = np.asarray(part["query_features"], dtype=np.float32)
        completed_steps = int(np.asarray(jax.device_get(state.step)))
        update_step = min(total_steps - 1, completed_steps)
        update_lr = learning_rate_at_step(config, update_step, total_steps)
        if args.method == "retrac":
            source_epoch = int(payload["epoch"])
            with checkpoints[checkpoint_index + 1].open("rb") as handle:
                target_epoch = int(pickle.load(handle)["epoch"])
            saved_replay_rngs = payload.get("replay_rngs")
            if saved_replay_rngs is None:
                raise RuntimeError(
                    f"{checkpoints[checkpoint_index]} predates multi-device ReTrac "
                    "replay metadata; retrain with the current 01_train_model.py"
                )
            replay_rngs = [jnp.asarray(value) for value in np.asarray(saved_replay_rngs)]
            device_count = len(replay_rngs)
            if int(config.batch_size) % device_count:
                raise RuntimeError("saved ReTrac device count does not divide the training batch")
            local_batch_size = int(config.batch_size) // device_count
            replay_bar = tqdm(total=(target_epoch - source_epoch) * steps_per_epoch, desc=f"ReTrac {source_epoch}->{target_epoch}", leave=False, unit="event-batch", dynamic_ncols=True)
            for epoch in range(source_epoch + 1, target_epoch + 1):
                order = np.arange(len(dataset), dtype=np.int64)
                np.random.default_rng(config.seed + epoch).shuffle(order)
                order = order[: steps_per_epoch * int(config.batch_size)]
                for batch_slot in range(steps_per_epoch):
                    indices = order[batch_slot * int(config.batch_size) : (batch_slot + 1) * int(config.batch_size)]
                    clean_all = dataset.images[indices]
                    timestep_parts = []
                    noise_parts = []
                    next_rngs = []
                    for device_slot, replay_rng in enumerate(replay_rngs):
                        next_rng, noise_rng, timestep_rng, _dropout_rng = jax.random.split(replay_rng, 4)
                        local_shape = clean_all[
                            device_slot * local_batch_size : (device_slot + 1) * local_batch_size
                        ].shape
                        timestep_parts.append(
                            np.asarray(
                                jax.random.randint(
                                    timestep_rng, (local_batch_size,), 0, config.timesteps
                                )
                            )
                        )
                        noise_parts.append(
                            np.asarray(jax.random.normal(noise_rng, local_shape, dtype=jnp.float32))
                        )
                        next_rngs.append(next_rng)
                    replay_rngs = next_rngs
                    timestep_all = np.concatenate(timestep_parts)
                    noise_all = np.concatenate(noise_parts)
                    selected = [slot for slot, index in enumerate(indices) if int(index) in destination]
                    if selected:
                        selected_indices = indices[selected]
                        event_step = (epoch - 1) * steps_per_epoch + batch_slot
                        event_lr = learning_rate_at_step(config, event_step, total_steps)
                        for event_start in range(0, len(selected), args.batch_size):
                            event_end = min(event_start + args.batch_size, len(selected))
                            real_count = event_end - event_start
                            slots = selected[event_start:event_end]
                            event_indices = indices[slots]
                            clean_rows = list(clean_all[slots, None, ...])
                            condition_rows = list(dataset.labels[event_indices, None, ...])
                            timestep_rows = list(timestep_all[slots])
                            noise_rows = list(noise_all[slots, None, ...])
                            while len(clean_rows) < args.batch_size:
                                clean_rows.append(clean_rows[-1])
                                condition_rows.append(condition_rows[-1])
                                timestep_rows.append(timestep_rows[-1])
                                noise_rows.append(noise_rows[-1])
                            event_timesteps = np.asarray(timestep_rows, dtype=np.int32)
                            event_alpha_bars = alpha_bars_all[event_timesteps]
                            projected = feature_function(
                                params,
                                opt_state,
                                jax.device_put(jnp.asarray(np.stack(clean_rows)), device),
                                jax.device_put(jnp.asarray(np.stack(condition_rows)), device),
                                jax.device_put(jnp.asarray(event_timesteps[:, None]), device),
                                jax.device_put(jnp.asarray(np.stack(noise_rows)), device),
                                jax.device_put(jnp.asarray(event_alpha_bars[:, None]), device),
                                jnp.asarray(update_lr, dtype=jnp.float32),
                                specs,
                            )
                            raw_projected, adamw_projected = (projected, None) if args.no_adamw else projected
                            raw_values = query_features @ np.asarray(jax.device_get(raw_projected))[:real_count].T
                            for local, dataset_index in enumerate(event_indices):
                                raw_scores[:, destination[int(dataset_index)]] += event_lr * raw_values[:, local]
                            if adamw_projected is not None:
                                assert adamw_scores is not None
                                values = query_features @ np.asarray(jax.device_get(adamw_projected))[:real_count].T
                                for local, dataset_index in enumerate(event_indices):
                                    adamw_scores[:, destination[int(dataset_index)]] += event_lr * values[:, local]
                    replay_bar.update(1)
            replay_bar.close()
            progress.update(math.ceil(len(train_indices) / args.batch_size))
        else:
            timesteps = np.asarray(part["timesteps"], dtype=np.int32)
            alpha_bars = alpha_bars_all[timesteps]
            for start in range(0, len(train_indices), args.batch_size):
                end = min(start + args.batch_size, len(train_indices))
                real_indices = train_indices[start:end]
                real_count = len(real_indices)
                indices = list(map(int, real_indices))
                while len(indices) < args.batch_size:
                    indices.append(indices[-1])
                indices_array = np.asarray(indices, dtype=np.int64)
                clean = dataset.images[indices_array, None, ...]
                condition = dataset.labels[indices_array, None, ...]
                noises = []
                for dataset_index in indices_array:
                    rows = []
                    for timestep in timesteps:
                        key = jax.random.PRNGKey(model_seed + 487_301)
                        for value in (int(dataset_index), checkpoint_index, int(timestep)):
                            key = jax.random.fold_in(key, value)
                        rows.append(np.asarray(jax.random.normal(key, clean[0].shape, dtype=jnp.float32)))
                    noises.append(np.stack(rows))
                projected = feature_function(
                    params,
                    opt_state,
                    jax.device_put(jnp.asarray(clean), device),
                    jax.device_put(jnp.asarray(condition), device),
                    jax.device_put(jnp.asarray(timesteps), device),
                    jax.device_put(jnp.asarray(np.stack(noises)), device),
                    jax.device_put(jnp.asarray(alpha_bars), device),
                    jnp.asarray(update_lr, dtype=jnp.float32),
                    specs,
                )
                raw_projected, adamw_projected = (projected, None) if args.no_adamw else projected
                raw_host = np.asarray(jax.device_get(raw_projected))[:real_count]
                query = query_features
                if args.method == "gas":
                    query = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
                    raw_host = raw_host / np.maximum(np.linalg.norm(raw_host, axis=1, keepdims=True), 1e-12)
                checkpoint_lr = learning_rate_at_step(config, max(0, completed_steps - 1), total_steps)
                raw_scores[:, start:end] += checkpoint_lr * (query @ raw_host.T)
                if adamw_projected is not None:
                    adamw_host = np.asarray(jax.device_get(adamw_projected))[:real_count]
                    if args.method == "gas":
                        adamw_host = adamw_host / np.maximum(np.linalg.norm(adamw_host, axis=1, keepdims=True), 1e-12)
                    assert adamw_scores is not None
                    adamw_scores[:, start:end] += checkpoint_lr * (query @ adamw_host.T)
                progress.update(1)
    progress.close()
    metadata = {
        "format_version": 2,
        "experiment": args.experiment,
        "method": args.method,
        "tag": tag,
        "normalization": manifest["query_reduction"],
        "score_normalization": manifest["score_normalization"],
        "projection": "countsketch",
        "projection_dim": args.projection_dim,
        "timesteps": spec.timestep_count,
        "checkpoint_pairs": len(parts),
        "query_manifest_sha256": hashlib.sha256(args.query_manifest.read_bytes()).hexdigest(),
        "scope": args.scope,
        "scope_size": len(scope_indices),
        "model_seed": model_seed,
        "mc_samples": spec.mc_samples,
        "mc_pairing": "method_specific_deterministic",
        "reduction": "linear",
        "learning_rate_weighted": True,
        "parameter_sources": ["raw_gradient"] if args.no_adamw else ["raw_gradient", "adamw_full"],
        "adamw_enabled": not args.no_adamw,
        "adamw_semantics": None if args.no_adamw else "saved-state next AdamW direction; LR applied once",
        "adamw_score_multiplier": None if args.no_adamw else -1,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "batch_size": args.batch_size,
    }
    save_shard(
        args.output_dir, args.shard_index, raw_scores=raw_scores,
        adamw_scores=orient_adamw_scores(args.method, adamw_scores),
        train_indices=train_indices, query_ids=query_ids, metadata=metadata,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
