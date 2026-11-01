from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np

PATTERN = re.compile(r"seed_-?\d+_epoch_(\d+)\.ckpt$")


def simple_loss_timesteps(experiment: str, total: int) -> np.ndarray:
    if experiment not in ("shape3d", "celeba"):
        raise ValueError(f"unsupported JAX experiment: {experiment}")
    count = 100 if experiment == "shape3d" else 50
    return np.linspace(total - 1, 0, count, dtype=np.int32)


def simple_loss_probe_seed(experiment: str, query_id: int) -> int | None:
    if experiment == "shape3d":
        return None
    if experiment == "celeba":
        return 2026 + 100_003 * int(query_id)
    raise ValueError(f"unsupported JAX experiment: {experiment}")


def _final_checkpoint(root: Path) -> Path:
    found = [(int(match.group(1)), path) for path in root.glob("seed_*_epoch_*.ckpt") if (match := PATTERN.match(path.name))]
    if not found:
        raise FileNotFoundError(f"no JAX checkpoint under {root}")
    return max(found)[1]


def evaluate(
    experiment: str,
    model_dir: Path,
    reference_dir: Path,
    query_manifest: Path,
    dataset: Path,
    output: Path,
) -> None:
    import jax
    import jax.numpy as jnp
    from tqdm.auto import tqdm

    from pint.training.backends.jax_pixel import (
        TrainConfig, _restore_checkpoint, build_model, choose_devices,
        create_train_state, make_diffusion_schedule,
    )

    target_path = _final_checkpoint(model_dir)
    reference_path = _final_checkpoint(reference_dir)
    with target_path.open("rb") as handle:
        payload = pickle.load(handle)
    config = TrainConfig(**{
        **payload["config"], "data_root": str(dataset), "prefer_device": "gpu",
        "use_data_parallel": False, "num_devices": 1,
    })
    device = choose_devices("gpu")[0]
    model = build_model(config)
    template = create_train_state(config, model, jax.random.PRNGKey(config.seed), device)
    target_state, _ = _restore_checkpoint(str(target_path), template)
    reference_state, _ = _restore_checkpoint(str(reference_path), template)
    target_params = jax.device_put(target_state.ema_params, device)
    reference_params = jax.device_put(reference_state.ema_params, device)
    @jax.jit
    def predicted(params, xt, timesteps, condition):
        return model.apply({"params": params}, xt, timesteps, condition, train=False)

    schedule = make_diffusion_schedule(
        config.timesteps, config.beta_start, config.beta_end
    )
    alpha_bars = np.asarray(schedule.alphas_cumprod, dtype=np.float32)
    simple_timesteps = simple_loss_timesteps(experiment, config.timesteps)
    shape3d_noise_seeds = np.arange(
        100_000, 100_000 + len(simple_timesteps), dtype=np.int32
    )

    manifest = json.loads(query_manifest.read_text(encoding="utf-8"))
    records = manifest["queries"] if isinstance(manifest, dict) else manifest
    labels = list(manifest.get("label_names", [])) if isinstance(manifest, dict) else []
    label_to_index = {name: index for index, name in enumerate(labels)}
    simple_values, deviation_values, query_ids = [], [], []
    for record in tqdm(records, desc="Evaluating LDS true f", unit="query", leave=False):
        query_dir = Path(record["dir"])
        trajectory = np.asarray(np.load(query_dir / "trajectory_xt.npy"), dtype=np.float32)[:, 0]
        timesteps = np.asarray(np.load(query_dir / "trajectory_t.npy"), dtype=np.int32)
        if experiment == "celeba":
            condition = np.asarray(record["labels"], dtype=np.float32)[None, :]
        else:
            condition = np.zeros((1, config.num_classes), dtype=np.float32)
            for label in record["labels"]:
                condition[0, label_to_index[str(label)]] = 1.0
        repeated = np.repeat(condition, len(timesteps), axis=0)
        target_prediction = predicted(target_params, trajectory, timesteps, repeated)
        reference_prediction = predicted(reference_params, trajectory, timesteps, repeated)
        delta = np.asarray(target_prediction - reference_prediction, dtype=np.float64)
        # Refined JAX LDS uses snapshot_mean: mean over prediction dimensions,
        # followed by mean over the saved reference-trajectory snapshots.
        deviation_values.append(float(np.mean(delta * delta)))
        endpoint = np.asarray(
            np.load(query_dir / "final_state.npy"), dtype=np.float32
        )
        probe_seed = simple_loss_probe_seed(experiment, int(record["query_id"]))
        if experiment == "shape3d":
            noise_keys = [
                jax.random.PRNGKey(int(seed)) for seed in shape3d_noise_seeds
            ]
        else:
            noise_keys = jax.random.split(
                jax.random.PRNGKey(int(probe_seed)), len(simple_timesteps)
            )
        noises = np.stack([
            np.asarray(
                jax.random.normal(key, endpoint.shape, dtype=jnp.float32)
            )
            for key in noise_keys
        ])
        clean = np.repeat(endpoint[None, ...], len(simple_timesteps), axis=0)
        noisy = (
            np.sqrt(alpha_bars[simple_timesteps])[:, None, None, None, None] * clean
            + np.sqrt(1.0 - alpha_bars[simple_timesteps])[:, None, None, None, None]
            * noises
        )
        simple_condition = np.repeat(condition[None, ...], len(simple_timesteps), axis=0)
        simple_prediction = predicted(
            target_params,
            noisy.reshape((-1, *endpoint.shape[1:])),
            np.repeat(simple_timesteps, endpoint.shape[0]),
            simple_condition.reshape((-1, *condition.shape[1:])),
        )
        simple_values.append(float(np.mean(
            (np.asarray(simple_prediction) - noises.reshape(np.asarray(simple_prediction).shape)) ** 2
        )))
        query_ids.append(int(record["query_id"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        query_ids=np.asarray(query_ids, dtype=np.int32),
        simple_loss=np.asarray(simple_values, dtype=np.float64),
        trajectory_deviation=np.asarray(deviation_values, dtype=np.float64),
        parameter_source=np.asarray("ema"),
        simple_loss_timesteps=np.asarray(simple_timesteps, dtype=np.int32),
        simple_loss_mc=np.asarray(1, dtype=np.int32),
        simple_loss_probe_scheme=np.asarray(
            "paired_seeds_100000_100099"
            if experiment == "shape3d"
            else "query_seed_2026_plus_100003_query_id"
        ),
        trajectory_reduction=np.asarray("mean_snapshot_mse"),
    )
