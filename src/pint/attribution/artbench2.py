from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np

from pint.attribution.common import (
    attribution_metadata,
    datapoint_shard,
    finish_timestamp_accumulator,
    load_query_parts,
    open_timestamp_accumulator,
    save_shard,
)
from pint.attribution.torch_adamw import adamw_full_direction_batch
from pint.query_gradients.artbench2 import (
    _build_unet,
    _checkpoints,
    _load_lora_checkpoint,
)
from pint.query_gradients.common import (
    artbench_projection_seed_parts, stable_torch_seed,
)
from pint.query_gradients.torch_projector import CountSketchProjector
from pint.training.model_versions import ARTBENCH_MODEL_ID, ARTBENCH_MODEL_REVISION


class LinearGradSampleCapture:
    """Exact batched per-sample gradients for LoRA-owned Linear parameters."""

    def __init__(self, root, parameters) -> None:
        import torch

        self.parameters = list(parameters)
        self.parameter_slots = {id(parameter): slot for slot, parameter in enumerate(self.parameters)}
        owners = set()
        modules = []
        for module in root.modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            relevant = False
            for _, parameter in module.named_parameters(recurse=False):
                if id(parameter) in self.parameter_slots:
                    owners.add(id(parameter))
                    relevant = True
            if relevant:
                modules.append(module)
        missing = [slot for slot, parameter in enumerate(self.parameters) if id(parameter) not in owners]
        if missing:
            raise ValueError(f"linear-hooks cannot locate LoRA parameter slots {missing[:8]}")
        self.enabled = False
        self.batch_size = 0
        self.samples = {}
        self.handles = [module.register_forward_hook(self._forward_hook) for module in modules]

    def start(self, batch_size: int) -> None:
        if self.enabled:
            raise RuntimeError("linear gradient capture is already active")
        self.batch_size = batch_size
        self.samples = {}
        self.enabled = True

    def _forward_hook(self, module, inputs, output) -> None:
        if not self.enabled:
            return
        activation = inputs[0].detach()
        if activation.shape[0] != self.batch_size or output.shape[0] != self.batch_size:
            raise ValueError("captured Linear does not preserve the batch axis")

        def capture(gradient_output):
            inputs_flat = activation.float().reshape(self.batch_size, -1, activation.shape[-1])
            outputs_flat = gradient_output.detach().float().reshape(
                self.batch_size, -1, gradient_output.shape[-1]
            )
            for name, parameter in module.named_parameters(recurse=False):
                if id(parameter) not in self.parameter_slots:
                    continue
                if name == "weight":
                    value = __import__("torch").bmm(outputs_flat.transpose(1, 2), inputs_flat)
                elif name == "bias":
                    value = outputs_flat.sum(dim=1)
                else:
                    raise ValueError(f"unsupported captured Linear parameter {name!r}")
                previous = self.samples.get(id(parameter))
                self.samples[id(parameter)] = value if previous is None else previous + value

        output.register_hook(capture)

    def gradients(self, losses):
        import torch

        if losses.shape != (self.batch_size,):
            raise ValueError("linear-hooks requires one scalar loss per example")
        try:
            aggregate = torch.autograd.grad(losses.sum(), self.parameters, allow_unused=True)
        finally:
            self.enabled = False
        per_sample = [
            self.samples.get(
                id(parameter),
                torch.zeros(
                    (self.batch_size,) + tuple(parameter.shape),
                    device=parameter.device,
                    dtype=torch.float32,
                ),
            )
            for parameter in self.parameters
        ]
        difference = torch.zeros((), device=losses.device)
        reference = torch.zeros((), device=losses.device)
        for parameter, samples, total in zip(self.parameters, per_sample, aggregate):
            expected = torch.zeros_like(parameter) if total is None else total.detach()
            actual = samples.sum(dim=0)
            difference.add_((actual.float() - expected.float()).square().sum())
            reference.add_(expected.float().square().sum())
        relative = float((difference.sqrt() / reference.sqrt().clamp_min(1e-30)).cpu())
        if relative > 5e-3:
            raise RuntimeError(
                "linear-hooks per-sample gradients failed the aggregate-gradient "
                f"check: relative_l2={relative:.8e}"
            )
        return per_sample

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def _checkpoint_optimizer_state(checkpoint: Path):
    import torch

    candidates = sorted(checkpoint.glob("optimizer*.bin")) + sorted(checkpoint.glob("optimizer*.pt"))
    if not candidates:
        raise FileNotFoundError(f"optimizer state is missing from {checkpoint}")
    state = torch.load(candidates[0], map_location="cpu")
    return state


def _checkpoint_learning_rate(checkpoint: Path) -> float:
    state = _checkpoint_optimizer_state(checkpoint)
    rates = {float(group["lr"]) for group in state["param_groups"]}
    if len(rates) != 1:
        raise ValueError(f"expected one checkpoint learning rate, found {sorted(rates)}")
    return rates.pop()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute datapoint-sharded ArtBench-2 PINT attribution")
    parser.add_argument("--experiment", choices=("artbench2",), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--query-gradient-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", default="all")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--no-adamw", action="store_true")
    parser.add_argument("--model", default=ARTBENCH_MODEL_ID)
    parser.add_argument("--model-revision", default=ARTBENCH_MODEL_REVISION)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args(argv)
    if args.scope != "all":
        parser.error("ArtBench-2 attribution currently uses scope 'all'")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    try:
        import pandas as pd
        import torch
        import torch.nn.functional as functional
        from diffusers import AutoencoderKL, DDPMScheduler
        from PIL import Image
        from torchvision import transforms
        from tqdm.auto import tqdm
        from transformers import CLIPTextModel, CLIPTokenizer
    except ImportError as exc:
        raise SystemExit("ArtBench-2 attribution requires train-artbench") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("ArtBench-2 attribution requires an NVIDIA GPU")
    device = torch.device("cuda:0")

    training_config_path = args.model_dir / "training_config.json"
    training_config = json.loads(training_config_path.read_text(encoding="utf-8"))
    model_seed = int(training_config["seed"])
    rank = int(training_config.get("rank", 128))
    if args.model == ARTBENCH_MODEL_ID:
        args.model = training_config.get("model", args.model)
    if args.model_revision == ARTBENCH_MODEL_REVISION:
        args.model_revision = training_config.get("model_revision", args.model_revision)
    metadata_csv = Path(training_config["metadata_csv"])
    image_root = Path(training_config["image_root"])
    index_path = Path(training_config["index_path"])
    resolution = int(training_config.get("resolution", 256))

    query_manifest, query_parts = load_query_parts(args.query_gradient_dir)
    checkpoints = _checkpoints(args.model_dir)
    if len(query_parts) != len(checkpoints) - 1:
        raise RuntimeError("query gradients and ArtBench-2 checkpoints have different pair counts")
    frame = pd.read_csv(metadata_csv)
    with index_path.open("rb") as handle:
        selected_rows = list(pickle.load(handle))
    selected_frame = frame.loc[selected_rows].reset_index(drop=False)
    scope_indices = np.arange(len(selected_frame), dtype=np.int64)
    owned = datapoint_shard(len(scope_indices), args.shard_index, args.shard_count)
    train_indices = scope_indices[list(owned)]
    if not len(train_indices):
        raise RuntimeError("more GPU shards than ArtBench-2 attribution datapoints")

    weight_dtype = torch.float16
    tokenizer = CLIPTokenizer.from_pretrained(
        args.model, subfolder="tokenizer", revision=args.model_revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.model,
        subfolder="text_encoder",
        revision=args.model_revision,
        torch_dtype=weight_dtype,
    ).to(device).eval()
    vae = AutoencoderKL.from_pretrained(
        args.model,
        subfolder="vae",
        revision=args.model_revision,
        torch_dtype=weight_dtype,
    ).to(device).eval()
    scheduler = DDPMScheduler.from_pretrained(
        args.model, subfolder="scheduler", revision=args.model_revision
    )
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    transform = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    labels = [str(selected_frame.iloc[int(index)]["label"]) for index in train_indices]
    hidden_by_label = {}
    with torch.no_grad():
        for label in sorted(set(labels)):
            prompt = f"a {label.replace('_', ' ').lower()} painting"
            tokens = tokenizer(
                prompt,
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            hidden_by_label[label] = text_encoder(tokens)[0].detach().cpu()
    latent_bank = []
    encode_progress = tqdm(
        range(0, len(train_indices), args.batch_size),
        desc=f"Encoding ArtBench-2 shard {args.shard_index + 1}/{args.shard_count}",
        unit="batch",
        dynamic_ncols=True,
    )
    with torch.no_grad():
        for start in encode_progress:
            batch_indices = train_indices[start : start + args.batch_size]
            pixels = []
            for index in batch_indices:
                row = selected_frame.iloc[int(index)]
                path = image_root / str(row["label"]) / str(row["name"])
                with Image.open(path) as image:
                    pixels.append(transform(image.convert("RGB")))
            pixel_batch = torch.stack(pixels).to(device=device, dtype=weight_dtype)
            latents = vae.encode(pixel_batch).latent_dist.mode() * vae.config.scaling_factor
            latent_bank.extend(latents.detach().cpu())

    unet, layers = _build_unet(args.model, args.model_revision, rank, device)
    capture = None
    query_ids = np.asarray(query_parts[0]["query_ids"], dtype=np.int32)
    timestep_count = len(query_manifest["timesteps"])
    accumulator_shape = (timestep_count, len(query_ids), len(train_indices))
    raw_path, raw_timestamps = open_timestamp_accumulator(
        args.output_dir, args.shard_index, "raw", accumulator_shape
    )
    adamw_path, adamw_timestamps = (
        (None, None) if args.no_adamw else
        open_timestamp_accumulator(
            args.output_dir, args.shard_index, "adamw", accumulator_shape
        )
    )
    total_units = len(query_parts) * math.ceil(len(train_indices) / args.batch_size) * len(query_manifest["timesteps"])
    progress = tqdm(
        total=total_units,
        desc=f"ArtBench-2 attribution shard {args.shard_index + 1}/{args.shard_count}",
        unit="batch-timestep",
        dynamic_ncols=True,
    )
    for checkpoint_index, part in enumerate(query_parts):
        if capture is not None:
            capture.close()
        layers = _load_lora_checkpoint(unet, layers, checkpoints[checkpoint_index])
        active = [parameter for parameter in layers.parameters() if parameter.requires_grad]
        capture = LinearGradSampleCapture(unet, active)
        optimizer = None
        if not args.no_adamw:
            optimizer = torch.optim.AdamW(
                active,
                lr=float(training_config.get("learning_rate", 1e-4)),
                weight_decay=float(training_config.get("weight_decay", 1e-2)),
            )
            optimizer.load_state_dict(_checkpoint_optimizer_state(checkpoints[checkpoint_index]))
        projector = CountSketchProjector(
            active, int(query_manifest["projection_dim"]),
            *artbench_projection_seed_parts(model_seed, checkpoint_index),
        )
        learning_rate = _checkpoint_learning_rate(checkpoints[checkpoint_index])
        timesteps = np.asarray(part["timesteps"], dtype=np.int32)
        features = np.asarray(part["query_features"], dtype=np.float32)
        if features.shape[:2] != (len(query_ids), len(timesteps)):
            raise ValueError("invalid ArtBench-2 query-gradient feature shape")
        query_features = torch.from_numpy(features).to(device)
        query_features = functional.normalize(query_features, dim=-1, eps=1e-12)
        noise_bank = []
        generator = torch.Generator(device=device)
        for timestep in timesteps:
            generator.manual_seed(
                stable_torch_seed(model_seed, checkpoint_index, int(timestep), "query_noise")
            )
            noise_bank.append(
                torch.randn(
                    (1, *latent_bank[0].shape),
                    generator=generator,
                    device=device,
                    dtype=weight_dtype,
                )
            )
        for destination_start in range(0, len(train_indices), args.batch_size):
            destination_end = min(destination_start + args.batch_size, len(train_indices))
            clean = torch.stack(latent_bank[destination_start:destination_end]).to(device)
            batch_labels = labels[destination_start:destination_end]
            hidden = torch.cat([hidden_by_label[label] for label in batch_labels]).to(device)
            for timestep_slot, (timestep, shared_noise) in enumerate(zip(timesteps, noise_bank)):
                noise = shared_noise.expand_as(clean)
                t = torch.full((len(clean),), int(timestep), device=device, dtype=torch.long)
                noisy = scheduler.add_noise(clean, noise, t)
                if scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif scheduler.config.prediction_type == "v_prediction":
                    target = scheduler.get_velocity(clean, noise, t)
                else:
                    raise ValueError(f"unsupported prediction type {scheduler.config.prediction_type}")
                capture.start(len(clean))
                prediction = unet(noisy, t, hidden).sample
                losses = functional.mse_loss(
                    prediction.float(), target.float(), reduction="none"
                ).mean(dim=(1, 2, 3))
                gradients = capture.gradients(losses)
                projected = projector.project_batch(gradients)
                projected = functional.normalize(projected, dim=-1, eps=1e-12)
                raw_values = query_features[:, timestep_slot] @ projected.T
                raw_timestamps[
                    timestep_slot, :, destination_start:destination_end
                ] += (
                    raw_values.detach().cpu().numpy()
                    * learning_rate / len(timesteps)
                )
                if optimizer is not None:
                    directions, reconstructed_lr = adamw_full_direction_batch(
                        active,
                        gradients,
                        optimizer,
                        max_grad_norm=float(training_config.get("max_grad_norm", 1.0)),
                    )
                    if not np.isclose(learning_rate, reconstructed_lr):
                        raise RuntimeError("optimizer LR differs from checkpoint LR")
                    assert adamw_timestamps is not None
                    adamw_projected = projector.project_batch(directions)
                    adamw_projected = functional.normalize(
                        adamw_projected, dim=-1, eps=1e-12
                    )
                    adamw_values = query_features[:, timestep_slot] @ adamw_projected.T
                    adamw_timestamps[
                        timestep_slot, :, destination_start:destination_end
                    ] += (
                        adamw_values.detach().cpu().numpy()
                        * learning_rate / len(timesteps)
                    )
                progress.update(1)
    progress.close()
    raw_scores = finish_timestamp_accumulator(raw_path, raw_timestamps)
    adamw_scores = (
        None if adamw_timestamps is None else
        finish_timestamp_accumulator(adamw_path, adamw_timestamps)
    )
    metadata = attribution_metadata(
        experiment="artbench2",
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
    metadata["per_example_backend"] = "linear_hooks"
    metadata["linear_hooks_aggregate_check"] = "relative_l2<=5e-3"
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
