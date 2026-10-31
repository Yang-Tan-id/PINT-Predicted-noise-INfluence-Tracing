from __future__ import annotations

import argparse
import json
import pickle
import re
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np

from pint.sampling.queries import celeba_queries, shape3d_queries


CHECKPOINT_PATTERN = re.compile(r"^seed_(?P<seed>-?\d+)_epoch_(?P<epoch>\d+)\.ckpt$")


def _final_checkpoint(model_dir: Path) -> Path:
    checkpoints = []
    for path in model_dir.glob("seed_*_epoch_*.ckpt"):
        match = CHECKPOINT_PATTERN.match(path.name)
        if match:
            checkpoints.append((int(match.group("seed")), int(match.group("epoch")), path))
    if not checkpoints:
        raise FileNotFoundError(
            f"No JAX checkpoint found under {model_dir}; run 01_train_model.py first"
        )
    seeds = {seed for seed, _, _ in checkpoints}
    if len(seeds) != 1:
        raise RuntimeError(
            f"Expected checkpoints from exactly one model seed under {model_dir}, "
            f"found seeds {sorted(seeds)}. Keep one model seed per training directory."
        )
    return max(checkpoints, key=lambda item: item[1])[2]


def _label_names(dataset: Path) -> list[str]:
    with np.load(dataset, allow_pickle=False) as payload:
        return [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in payload["label_names"].tolist()
        ]


def _save_image(array: np.ndarray, path: Path) -> None:
    from PIL import Image

    image = np.clip((array + 1.0) / 2.0, 0.0, 1.0)
    Image.fromarray(np.rint(image * 255).astype(np.uint8)).save(path)


def inference_timesteps(total_timesteps: int, num_inference_steps: int) -> list[int]:
    """Match the timestep grid used by the refined JAX sampler."""

    if num_inference_steps <= 0 or num_inference_steps > total_timesteps:
        raise ValueError(
            f"DDIM steps must be in [1, {total_timesteps}], got {num_inference_steps}"
        )
    if total_timesteps % num_inference_steps == 0:
        stride = total_timesteps // num_inference_steps
        return list(range(total_timesteps - 1, -1, -stride))
    return [
        int(round(value))
        for value in np.linspace(total_timesteps - 1, 0, num_inference_steps)
    ]


def default_ddim_steps(experiment: str) -> int:
    return 50 if experiment == "celeba" else 1000


def _sample_trajectory(state, model, schedule, config, seed, condition, steps, snapshots):
    import jax
    import jax.numpy as jnp

    timestep_grid = inference_timesteps(config.timesteps, steps)
    if len(set(timestep_grid)) != len(timestep_grid):
        raise ValueError("DDIM timestep grid contains duplicates")
    previous_timesteps = timestep_grid[1:] + [-1]
    t_sequence = jnp.asarray(timestep_grid, dtype=jnp.int32)
    previous_sequence = jnp.asarray(previous_timesteps, dtype=jnp.int32)
    shape = (1, config.image_size, config.image_size, config.in_channels)
    condition_array = None if condition is None else jnp.asarray(condition, dtype=jnp.float32)
    alphas_cumprod = schedule.alphas_cumprod

    @jax.jit
    def run(initial_rng):
        initial_x = jax.random.normal(initial_rng, shape)

        def body(carry, pair):
            x_t = carry
            timestep, previous_timestep = pair
            batch_t = jnp.full((shape[0],), timestep, dtype=jnp.int32)
            prediction = model.apply(
                {"params": state.ema_params},
                x_t,
                batch_t,
                condition_array if config.class_cond else None,
                train=False,
            )
            if config.predict_x0:
                x0_prediction = prediction
                epsilon = (
                    x_t - jnp.sqrt(alphas_cumprod[timestep]) * x0_prediction
                ) / jnp.sqrt(1.0 - alphas_cumprod[timestep])
            else:
                epsilon = prediction
                x0_prediction = (
                    x_t - jnp.sqrt(1.0 - alphas_cumprod[timestep]) * epsilon
                ) / jnp.sqrt(alphas_cumprod[timestep])
            previous_alpha_bar = jax.lax.cond(
                previous_timestep >= 0,
                lambda _: alphas_cumprod[jnp.maximum(previous_timestep, 0)],
                lambda _: jnp.asarray(1.0, dtype=alphas_cumprod.dtype),
                operand=None,
            )
            next_x = (
                jnp.sqrt(previous_alpha_bar) * x0_prediction
                + jnp.sqrt(jnp.maximum(1.0 - previous_alpha_bar, 0.0)) * epsilon
            )
            return next_x, x_t

        final_x, states = jax.lax.scan(
            body, initial_x, (t_sequence, previous_sequence)
        )
        return final_x, states

    final_state, all_states = run(jax.random.PRNGKey(seed))
    positions = np.linspace(0, steps - 1, snapshots, dtype=np.int32)
    states = np.asarray(all_states)[positions]
    timesteps = np.asarray(
        [timestep_grid[int(position)] for position in positions], dtype=np.int32
    )
    return np.asarray(final_state), states, timesteps


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sample JAX pixel-DDPM queries")
    parser.add_argument("--experiment", required=True, choices=("shape3d", "celeba"))
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--partition", type=Path)
    parser.add_argument("--attributes", type=Path)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ddim-steps", type=int)
    parser.add_argument("--trajectory-snapshots", type=int)
    parser.add_argument("--device", choices=("gpu", "cpu", "auto"), default="gpu")
    args = parser.parse_args(argv)
    if args.ddim_steps is None:
        args.ddim_steps = default_ddim_steps(args.experiment)
    if args.trajectory_snapshots is None:
        args.trajectory_snapshots = min(100, args.ddim_steps)
    if args.num_queries <= 0:
        parser.error("--num-queries must be positive")
    if not 1 <= args.trajectory_snapshots <= args.ddim_steps:
        parser.error("--trajectory-snapshots must be between 1 and --ddim-steps")

    try:
        import jax
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise SystemExit("Shape3D/CelebA sampling requires the train-jax dependencies") from exc

    from pint.datasets.celeba import read_attributes, read_partition
    from pint.datasets.paths import data_root
    from pint.training.backends.jax_pixel import (
        TrainConfig,
        _restore_checkpoint,
        build_model,
        choose_devices,
        create_train_state,
        make_diffusion_schedule,
    )

    if args.dataset is None:
        size = "20000" if args.experiment == "shape3d" else "10000"
        args.dataset = data_root() / args.experiment / size / "dataset.npz"
    if not args.dataset.is_file():
        raise FileNotFoundError(f"Prepared dataset not found: {args.dataset}")
    checkpoint_path = _final_checkpoint(args.model_dir)
    with checkpoint_path.open("rb") as handle:
        payload = pickle.load(handle)
    config = TrainConfig(
        **{
            **payload["config"],
            "data_root": str(args.dataset),
            "prefer_device": args.device,
            "use_data_parallel": False,
            "num_devices": 1,
        }
    )
    device = choose_devices(args.device)[0]
    model = build_model(config)
    template = create_train_state(
        config, model, jax.random.PRNGKey(config.seed), device
    )
    state, epoch = _restore_checkpoint(str(checkpoint_path), template)
    schedule = make_diffusion_schedule(
        config.timesteps, config.beta_start, config.beta_end
    )
    labels = _label_names(args.dataset)
    label_to_index = {label: index for index, label in enumerate(labels)}
    if args.experiment == "shape3d":
        records = shape3d_queries(args.num_queries, args.seed)
    else:
        args.partition = args.partition or (
            data_root() / "raw" / "celeba" / "list_eval_partition.txt"
        )
        args.attributes = args.attributes or (
            data_root() / "raw" / "celeba" / "list_attr_celeba.txt"
        )
        for source in (args.partition, args.attributes):
            if not source.is_file():
                raise FileNotFoundError(f"CelebA query source not found: {source}")
        attribute_names, attribute_rows = read_attributes(args.attributes)
        if list(attribute_names) != labels:
            raise ValueError(
                "CelebA Attr40 order differs between the raw attribute file and "
                "the prepared training dataset; refusing to apply labels to the "
                "wrong conditioning dimensions"
            )
        records = celeba_queries(
            args.num_queries,
            attribute_names,
            attribute_rows,
            read_partition(args.partition),
            args.seed,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    width = max(2, len(str(args.num_queries - 1)))
    manifest_records = []

    for record in tqdm(records, desc=f"Sampling {args.experiment} queries", unit="query"):
        condition = None
        if config.class_cond:
            if args.experiment == "celeba":
                condition = np.asarray(record["labels"], dtype=np.float32)[None, :]
            else:
                condition = np.zeros((1, config.num_classes), dtype=np.float32)
                for label in record["labels"]:
                    condition[0, label_to_index[str(label)]] = 1.0
        final_state, trajectory, timesteps = _sample_trajectory(
            state,
            model,
            schedule,
            config,
            int(record["initial_seed"]),
            condition,
            args.ddim_steps,
            args.trajectory_snapshots,
        )
        query_dir = args.output_dir / f"q{int(record['query_id']):0{width}d}"
        query_dir.mkdir(parents=True, exist_ok=True)
        np.save(query_dir / "trajectory_xt.npy", trajectory)
        np.save(query_dir / "trajectory_t.npy", timesteps)
        np.save(query_dir / "final_state.npy", final_state)
        _save_image(final_state[0], query_dir / "final.png")
        saved_record = {
            **record,
            "dir": str(query_dir),
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": epoch,
            "model_seed": config.seed,
            "sampler": "ddim_eta0",
        }
        manifest_records.append(saved_record)
        (query_dir / "query.json").write_text(
            json.dumps(saved_record, indent=2) + "\n", encoding="utf-8"
        )

    manifest = {
        "format_version": 1,
        "experiment": args.experiment,
        "sampler": "ddim_eta0",
        "num_inference_steps": args.ddim_steps,
        "checkpoint": str(checkpoint_path),
        "model_seed": config.seed,
        "checkpoint_config": asdict(config),
        "label_names": labels,
        "queries": manifest_records,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(records)} queries to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
