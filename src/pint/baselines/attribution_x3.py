from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from pint.attribution.common import datapoint_shard, save_shard
from pint.attribution.torch_adamw import adamw_full_direction_batch
from pint.baselines.common import baseline_spec, load_query_run, run_tag, validate_query_source
from pint.query_gradients.common import stable_torch_seed
from pint.query_gradients.torch_projector import CountSketchProjector
from pint.query_gradients.x3 import _build_model, _checkpoints
from pint.training.backends.x3_dataset import ColorGridDataset
from pint.training.x3_training import _learning_rate


def _metadata(args, manifest, model_seed, scope_size, tag):
    return {
        "format_version": 2,
        "experiment": "x3",
        "method": args.method,
        "tag": tag,
        "normalization": manifest["query_reduction"],
        "score_normalization": manifest["score_normalization"],
        "projection": "countsketch",
        "projection_dim": int(manifest["projection_dim"]),
        "timesteps": int(manifest["timestep_count"]),
        "checkpoint_pairs": int(manifest["source_checkpoint_count"]),
        "query_manifest_sha256": hashlib.sha256(args.query_manifest.read_bytes()).hexdigest(),
        "scope": args.scope,
        "scope_size": scope_size,
        "model_seed": model_seed,
        "mc_samples": int(manifest["mc_samples"]),
        "mc_pairing": "method_specific_deterministic",
        "reduction": "linear",
        "learning_rate_weighted": True,
        "parameter_sources": ["raw_gradient"] if args.no_adamw else ["raw_gradient", "adamw_full"],
        "adamw_enabled": not args.no_adamw,
        "adamw_semantics": None if args.no_adamw else "saved-state next AdamW direction; LR applied once",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "batch_size": args.batch_size,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("x3",), required=True)
    parser.add_argument("--method", choices=("gas", "tracincp", "retrac"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--query-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--projection-dim", type=int, default=4096)
    parser.add_argument("--scope", default="all")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-adamw", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args(argv)
    if args.scope != "all":
        parser.error("x3 baselines use scope 'all'")
    try:
        import torch
        import torch.nn.functional as functional
        from torch.func import functional_call, grad, vmap
        from torch.utils.data import DataLoader, Dataset
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise SystemExit("x3 baselines require train-x3") from exc
    device = torch.device("cuda:0" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    spec = baseline_spec(args.method, "x3")
    tag = run_tag(spec, args.projection_dim)
    manifest, parts = load_query_run(args.query_dir, tag)
    validate_query_source(manifest, args.query_manifest)
    checkpoints = _checkpoints(args.model_dir)
    first_model, first_payload = _build_model(checkpoints[0], device)
    model_seed = int(first_payload["seed"])
    training_config = args.model_dir / "base" / "prompted" / "training_config.json"
    configured = json.loads(training_config.read_text(encoding="utf-8"))
    dataset = ColorGridDataset(str(configured["dataset"]), grid_size=3)
    all_indices = np.arange(len(dataset), dtype=np.int64)
    owned_range = datapoint_shard(len(all_indices), args.shard_index, args.shard_count)
    train_indices = all_indices[list(owned_range)]
    destination = {int(index): slot for slot, index in enumerate(train_indices)}
    query_ids = np.asarray(parts[0]["query_ids"], dtype=np.int32)
    raw_scores = np.zeros((len(query_ids), len(train_indices)), dtype=np.float64)
    adamw_scores = None if args.no_adamw else np.zeros_like(raw_scores)
    from pint.training.backends.x3_model import make_linear_schedule, q_sample

    schedule = make_linear_schedule(int(first_payload.get("T", 1000)), device=device)
    progress_total = len(parts) * max(1, math.ceil(len(train_indices) / args.batch_size))
    if args.method != "retrac":
        progress_total *= spec.timestep_count
    progress = tqdm(total=progress_total, desc=f"x3 {args.method} attribution", unit="batch", dynamic_ncols=True)

    def optimizer_for(model, payload):
        active = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if args.no_adamw:
            return active, None
        config = payload["config"]
        optimizer = torch.optim.AdamW(
            active,
            lr=float(payload["learning_rate_at_checkpoint"]),
            betas=(float(config["adam_b1"]), float(config["adam_b2"])),
            eps=float(config["adam_eps"]),
            weight_decay=float(config["weight_decay"]),
        )
        optimizer.load_state_dict(payload["optimizer_state"])
        return active, optimizer

    for checkpoint_index, part in enumerate(parts):
        model, payload = (first_model, first_payload) if checkpoint_index == 0 else _build_model(checkpoints[checkpoint_index], device)
        named = dict(model.named_parameters())
        active, optimizer = optimizer_for(model, payload)
        namespace = "diffusion_retrac_projection" if args.method == "retrac" else "endpoint_tracin_adamw_mc10_projection"
        projector = CountSketchProjector(active, args.projection_dim, model_seed, namespace, checkpoint_index)
        query_features = torch.from_numpy(np.asarray(part["query_features"], dtype=np.float32)).to(device)

        def loss_one(parameters, clean, condition, timestep, noise):
            t = timestep.reshape(1)
            prediction = functional_call(
                model,
                parameters,
                (q_sample(clean.unsqueeze(0), t, noise.unsqueeze(0), schedule), t, condition.unsqueeze(0)),
            )
            return functional.mse_loss(prediction, noise.unsqueeze(0))

        gradient_one = vmap(grad(loss_one), in_dims=(None, 0, 0, 0, 0))
        if args.method == "retrac":
            required = ("torch_rng_state", "cuda_rng_state_all")
            if any(key not in payload for key in required):
                raise RuntimeError(
                    f"{checkpoints[checkpoint_index]} predates ReTrac replay metadata; "
                    "retrain with the current 01_train_model.py"
                )

            class IndexedDataset(Dataset):
                def __len__(self):
                    return len(dataset)

                def __getitem__(self, index):
                    clean, condition = dataset[index]
                    return clean, condition, index

            torch.set_rng_state(payload["torch_rng_state"])
            if device.type == "cuda":
                states = payload["cuda_rng_state_all"]
                torch.cuda.set_rng_state(states[0], device)
            replay_loader = DataLoader(
                IndexedDataset(),
                batch_size=int(configured["batch_size"]),
                shuffle=True,
                num_workers=int(configured.get("num_workers", 0)),
                drop_last=False,
                pin_memory=device.type == "cuda",
            )
            source_epoch = int(payload["epoch"])
            target_epoch = int(torch.load(checkpoints[checkpoint_index + 1], map_location="cpu")["epoch"])
            steps_per_epoch = math.ceil(len(dataset) / int(configured["batch_size"]))
            total_steps = int(configured["epochs"]) * steps_per_epoch
            replay_bar = tqdm(
                total=(target_epoch - source_epoch) * len(replay_loader),
                desc=f"x3 ReTrac interval {source_epoch}->{target_epoch}",
                unit="event-batch",
                leave=False,
                dynamic_ncols=True,
            )
            for epoch in range(source_epoch + 1, target_epoch + 1):
                for batch_slot, (clean_all, condition_all, indices_all) in enumerate(replay_loader):
                    clean_all = clean_all.to(device)
                    condition_all = condition_all.to(device)
                    timestep_all = torch.randint(0, int(payload["T"]), (len(clean_all),), device=device)
                    noise_all = torch.randn_like(clean_all)
                    selected = [slot for slot, index in enumerate(indices_all.tolist()) if int(index) in destination]
                    if selected:
                        clean = clean_all[selected]
                        condition = condition_all[selected]
                        timestep = timestep_all[selected]
                        noise = noise_all[selected]
                        gradients = gradient_one(named, clean, condition, timestep, noise)
                        raw_batch = list(gradients.values())
                        full_norm = torch.sqrt(
                            sum(value.float().flatten(1).square().sum(dim=1) for value in raw_batch)
                        ).clamp_min(1e-12)
                        raw_normalized = [
                            value / full_norm.reshape((len(clean),) + (1,) * (value.ndim - 1))
                            for value in raw_batch
                        ]
                        raw_projected = projector.project_batch(raw_normalized)
                        event_step = (epoch - 1) * steps_per_epoch + batch_slot
                        event_lr = _learning_rate(
                            event_step,
                            total_steps,
                            float(configured["learning_rate"]),
                            float(configured["warmup_ratio"]),
                        )
                        dots = query_features @ raw_projected.T
                        for local, sample_slot in enumerate(selected):
                            raw_scores[:, destination[int(indices_all[sample_slot])]] += event_lr * dots[:, local].double().cpu().numpy()
                        if optimizer is not None:
                            directions, _ = adamw_full_direction_batch(
                                active, raw_batch, optimizer, max_grad_norm=float(configured["grad_clip_norm"])
                            )
                            direction_norm = torch.sqrt(
                                sum(value.float().flatten(1).square().sum(dim=1) for value in directions)
                            ).clamp_min(1e-12)
                            normalized_directions = [
                                value / direction_norm.reshape((len(clean),) + (1,) * (value.ndim - 1))
                                for value in directions
                            ]
                            projected = projector.project_batch(normalized_directions)
                            values = query_features @ projected.T
                            assert adamw_scores is not None
                            for local, sample_slot in enumerate(selected):
                                adamw_scores[:, destination[int(indices_all[sample_slot])]] += event_lr * values[:, local].double().cpu().numpy()
                    replay_bar.update(1)
            replay_bar.close()
            progress.update(max(1, math.ceil(len(train_indices) / args.batch_size)))
        else:
            timesteps = np.asarray(part["timesteps"], dtype=np.int32)
            for start in range(0, len(train_indices), args.batch_size):
                end = min(start + args.batch_size, len(train_indices))
                batch_indices = train_indices[start:end]
                examples = [dataset[int(index)] for index in batch_indices]
                clean = torch.stack([item[0] for item in examples]).to(device)
                condition = torch.stack([item[1] for item in examples]).to(device)
                for timestamp_slot, timestep in enumerate(timesteps):
                    noise_rows = []
                    noise_groups = {}
                    for dataset_index in batch_indices:
                        micro_start = (int(dataset_index) // 32) * 32
                        if micro_start not in noise_groups:
                            micro_count = min(32, len(dataset) - micro_start)
                            generator = torch.Generator(device=device)
                            generator.manual_seed(
                                stable_torch_seed(
                                    model_seed,
                                    "endpoint_tracin_train_mc10",
                                    checkpoint_index,
                                    timestamp_slot,
                                    micro_start,
                                )
                            )
                            noise_groups[micro_start] = torch.randn(
                                (micro_count, spec.mc_samples, *clean.shape[1:]),
                                generator=generator,
                                device=device,
                                dtype=clean.dtype,
                            )
                        noise_rows.append(
                            noise_groups[micro_start][int(dataset_index) - micro_start]
                        )
                    noises = torch.stack(noise_rows)
                    count = len(clean)
                    expanded_clean = clean[:, None].expand(count, spec.mc_samples, *clean.shape[1:]).reshape(-1, *clean.shape[1:])
                    expanded_condition = condition[:, None].expand(count, spec.mc_samples, condition.shape[-1]).reshape(-1, condition.shape[-1])
                    t = torch.full((count * spec.mc_samples,), int(timestep), device=device, dtype=torch.long)
                    gradients = gradient_one(named, expanded_clean, expanded_condition, t, noises.reshape(-1, *clean.shape[1:]))
                    raw_batch = [value.reshape(count, spec.mc_samples, *value.shape[1:]).mean(dim=1) for value in gradients.values()]
                    projected = projector.project_batch(raw_batch)
                    query_t = query_features[:, timestamp_slot]
                    if args.method == "gas":
                        query_t = query_t / query_t.norm(dim=1, keepdim=True).clamp_min(1e-12)
                        projected = projected / projected.norm(dim=1, keepdim=True).clamp_min(1e-12)
                    learning_rate = float(payload["learning_rate_at_checkpoint"])
                    raw_scores[:, start:end] += learning_rate * (query_t @ projected.T).double().cpu().numpy() / len(timesteps)
                    if optimizer is not None:
                        directions, reconstructed_lr = adamw_full_direction_batch(
                            active, raw_batch, optimizer, max_grad_norm=float(payload["config"]["grad_clip_norm"])
                        )
                        if not np.isclose(learning_rate, reconstructed_lr):
                            raise RuntimeError("checkpoint optimizer LR mismatch")
                        adamw_projected = projector.project_batch(directions)
                        if args.method == "gas":
                            adamw_projected = adamw_projected / adamw_projected.norm(dim=1, keepdim=True).clamp_min(1e-12)
                        assert adamw_scores is not None
                        adamw_scores[:, start:end] += learning_rate * (query_t @ adamw_projected.T).double().cpu().numpy() / len(timesteps)
                    progress.update(1)
    progress.close()
    save_shard(
        args.output_dir,
        args.shard_index,
        raw_scores=raw_scores,
        adamw_scores=adamw_scores,
        train_indices=train_indices,
        query_ids=query_ids,
        metadata=_metadata(args, manifest, model_seed, len(all_indices), tag),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
