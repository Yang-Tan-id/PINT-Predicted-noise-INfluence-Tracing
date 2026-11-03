from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import random
from pathlib import Path
from typing import Sequence

import numpy as np

from pint.attribution.artbench2 import LinearGradSampleCapture, _checkpoint_optimizer_state
from pint.attribution.common import datapoint_shard, save_shard
from pint.attribution.torch_adamw import adamw_full_direction_batch
from pint.baselines.artbench_projection import (
    ArtBenchProjector, artbench_projection_seed, artbench_seed,
)
from pint.baselines.common import (
    baseline_spec, load_query_run, orient_adamw_scores, run_tag,
    validate_query_source,
)
from pint.query_gradients.artbench2 import _build_unet, _checkpoints, _load_lora_checkpoint
from pint.training.model_versions import ARTBENCH_MODEL_ID, ARTBENCH_MODEL_REVISION


def _restore_rng(checkpoint: Path, device) -> None:
    import torch

    paths = sorted(checkpoint.glob("random_states*.pkl"))
    if not paths:
        raise FileNotFoundError(f"no random_states*.pkl in {checkpoint}")
    state = torch.load(paths[0], map_location="cpu")
    if "random_state" in state:
        random.setstate(state["random_state"])
    if "numpy_random_seed" in state:
        np.random.set_state(state["numpy_random_seed"])
    if "torch_manual_seed" in state:
        torch.set_rng_state(state["torch_manual_seed"])
    cuda = state.get("torch_cuda_manual_seed")
    if cuda is not None and device.type == "cuda":
        torch.cuda.set_rng_state(cuda[0] if isinstance(cuda, (list, tuple)) else cuda, device)


def _cosine_lr(base: float, step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return base * float(step) / max(1, warmup)
    progress = min(max(float(step - warmup) / max(1, total - warmup), 0.0), 1.0)
    return base * 0.5 * (1.0 + math.cos(math.pi * progress))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("artbench2",), required=True)
    parser.add_argument("--method", choices=("gas", "tracincp", "retrac"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--query-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--projection-dim", type=int, default=32768)
    parser.add_argument("--scope", default="all")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--no-adamw", action="store_true")
    parser.add_argument("--model", default=ARTBENCH_MODEL_ID)
    parser.add_argument("--model-revision", default=ARTBENCH_MODEL_REVISION)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args(argv)
    if args.scope != "all":
        parser.error("ArtBench-2 baselines use scope 'all'")
    try:
        import pandas as pd
        import torch
        import torch.nn.functional as functional
        from diffusers import AutoencoderKL, DDPMScheduler
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset
        from torchvision import transforms
        from tqdm.auto import tqdm
        from transformers import CLIPTextModel, CLIPTokenizer
    except ImportError as exc:
        raise SystemExit("ArtBench-2 baselines require train-artbench") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("ArtBench-2 baselines require an NVIDIA GPU")
    from pint.training.backends.artbench_lora import IndexedArtBenchDataset, set_attention_dropout

    device = torch.device("cuda:0")
    config = json.loads((args.model_dir / "training_config.json").read_text(encoding="utf-8"))
    if args.model == ARTBENCH_MODEL_ID:
        args.model = config.get("model", args.model)
    if args.model_revision == ARTBENCH_MODEL_REVISION:
        args.model_revision = config.get("model_revision", args.model_revision)
    model_seed = int(config["seed"])
    rank = int(config.get("rank", 128))
    spec = baseline_spec(args.method, "artbench2")
    tag = run_tag(spec, args.projection_dim)
    manifest, parts = load_query_run(args.query_dir, tag)
    validate_query_source(manifest, args.query_manifest)
    checkpoints = _checkpoints(args.model_dir)
    tokenizer = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer", revision=args.model_revision)
    text_encoder = CLIPTextModel.from_pretrained(
        args.model, subfolder="text_encoder", revision=args.model_revision, torch_dtype=torch.float16
    ).to(device).eval()
    vae = AutoencoderKL.from_pretrained(
        args.model, subfolder="vae", revision=args.model_revision, torch_dtype=torch.float16
    ).to(device).eval()
    scheduler = DDPMScheduler.from_pretrained(args.model, subfolder="scheduler", revision=args.model_revision)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    frame = pd.read_csv(Path(config["metadata_csv"]))
    with Path(config["index_path"]).open("rb") as handle:
        selected_rows = list(pickle.load(handle))
    selected_frame = frame.loc[selected_rows].reset_index(drop=False)
    all_positions = np.arange(len(selected_frame), dtype=np.int64)
    owned = datapoint_shard(len(all_positions), args.shard_index, args.shard_count)
    train_indices = all_positions[list(owned)]
    destination = {int(index): slot for slot, index in enumerate(train_indices)}
    query_ids = np.asarray(parts[0]["query_ids"], dtype=np.int32)
    raw_scores = np.zeros((len(query_ids), len(train_indices)), dtype=np.float64)
    adamw_scores = None if args.no_adamw else np.zeros_like(raw_scores)
    unet, layers = _build_unet(args.model, args.model_revision, rank, device)
    deterministic_transform = transforms.Compose(
        [
            transforms.Resize(int(config.get("resolution", 256)), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(int(config.get("resolution", 256))),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    progress = tqdm(total=len(parts) * max(1, math.ceil(len(train_indices) / args.batch_size)), desc=f"ArtBench-2 {args.method} attribution", unit="batch", dynamic_ncols=True)
    for checkpoint_index, part in enumerate(parts):
        layers = _load_lora_checkpoint(unet, layers, checkpoints[checkpoint_index])
        active = [parameter for parameter in layers.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(active, lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
        optimizer.load_state_dict(_checkpoint_optimizer_state(checkpoints[checkpoint_index]))
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        checkpoint_lr = float(optimizer.param_groups[0]["lr"])
        projector = ArtBenchProjector(
            args.projection_dim, artbench_projection_seed(checkpoint_index)
        )
        query_features = torch.from_numpy(np.asarray(part["query_features"], dtype=np.float32)).to(device)
        if args.method == "retrac":
            training_dataset = IndexedArtBenchDataset(
                Path(config["metadata_csv"]),
                Path(config["image_root"]),
                Path(config["index_path"]),
                tokenizer,
                int(config.get("resolution", 256)),
            )

            class WithPosition(Dataset):
                def __len__(self):
                    return len(training_dataset)

                def __getitem__(self, index):
                    item = training_dataset[index]
                    item["position"] = index
                    return item

            _restore_rng(checkpoints[checkpoint_index], device)
            loader = DataLoader(
                WithPosition(),
                batch_size=int(config["batch_size"]),
                shuffle=True,
                num_workers=int(config["num_workers"]),
                pin_memory=True,
            )
            current_state = json.loads((checkpoints[checkpoint_index] / "training_state.json").read_text())
            next_state = json.loads((checkpoints[checkpoint_index + 1] / "training_state.json").read_text())
            start_epoch, end_epoch = int(current_state["completed_epoch"]), int(next_state["completed_epoch"])
            updates_per_epoch = math.ceil(len(loader) / int(config["gradient_accumulation_steps"]))
            total_steps = updates_per_epoch * int(config["epochs"])
            warmup_steps = math.ceil(total_steps * 0.1)
            replay_bar = tqdm(total=(end_epoch - start_epoch) * len(loader), desc=f"ReTrac {start_epoch}->{end_epoch}", unit="event-batch", leave=False, dynamic_ncols=True)
            for epoch in range(start_epoch, end_epoch):
                for batch_index, batch in enumerate(loader):
                    positions = [int(value) for value in batch["position"]]
                    pixels = batch["pixel_values"].to(device, dtype=torch.float16)
                    input_ids = batch["input_ids"].to(device)
                    # Consume the exact training RNG calls. ReTrac batch1 uses
                    # the resulting latent/noise/timestep but the clean prompt.
                    drop = torch.rand(input_ids.shape[0], device=device) < float(config["conditioning_dropout"])
                    dropped_ids = input_ids.clone()
                    empty = tokenizer("", max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids[0].to(device)
                    dropped_ids[drop] = empty
                    with torch.no_grad():
                        latents = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
                        noises = torch.randn_like(latents)
                        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (len(latents),), device=device).long()
                        replay_hidden = text_encoder(dropped_ids)[0]
                        clean_hidden = text_encoder(input_ids)[0]
                    # Consume the original attention-dropout RNG before the
                    # independent batch1 gradients, exactly as the established
                    # ArtBench-2 ReTrac implementation does.
                    set_attention_dropout(unet, float(config["attention_dropout"]))
                    unet.train()
                    with torch.no_grad():
                        unet(
                            scheduler.add_noise(latents, noises, timesteps),
                            timesteps,
                            replay_hidden,
                        )
                    set_attention_dropout(unet, 0.0)
                    unet.eval()
                    event_step = epoch * updates_per_epoch + batch_index // int(config["gradient_accumulation_steps"])
                    event_lr = _cosine_lr(float(config["learning_rate"]), event_step, total_steps, warmup_steps)
                    for sample_slot, position in enumerate(positions):
                        if position not in destination:
                            continue
                        t = timesteps[sample_slot : sample_slot + 1]
                        prediction = unet(
                            scheduler.add_noise(latents[sample_slot : sample_slot + 1], noises[sample_slot : sample_slot + 1], t),
                            t,
                            clean_hidden[sample_slot : sample_slot + 1],
                        ).sample
                        loss = functional.mse_loss(prediction.float(), noises[sample_slot : sample_slot + 1].float())
                        gradients = torch.autograd.grad(loss, active, allow_unused=True)
                        gradients = [torch.zeros_like(p) if g is None else g for p, g in zip(active, gradients)]
                        raw_feature = projector.project(gradients, full_l2_normalize=True)
                        raw_scores[:, destination[position]] += event_lr * (query_features @ raw_feature).double().cpu().numpy()
                        if not args.no_adamw:
                            batched = [value.unsqueeze(0) for value in gradients]
                            directions, _ = adamw_full_direction_batch(active, batched, optimizer, max_grad_norm=1.0)
                            adamw_feature = projector.project([value[0] for value in directions], full_l2_normalize=True)
                            assert adamw_scores is not None
                            adamw_scores[:, destination[position]] += event_lr * (query_features @ adamw_feature).double().cpu().numpy()
                    replay_bar.update(1)
            replay_bar.close()
            progress.update(max(1, math.ceil(len(train_indices) / args.batch_size)))
        else:
            timesteps = [int(value) for value in np.asarray(part["timesteps"])]
            set_attention_dropout(unet, 0.0)
            capture = LinearGradSampleCapture(unet, active)
            for start in range(0, len(train_indices), args.batch_size):
                end = min(start + args.batch_size, len(train_indices))
                positions = train_indices[start:end]
                pixel_rows = []
                prompts = []
                for position in positions:
                    row = selected_frame.iloc[int(position)]
                    with Image.open(Path(config["image_root"]) / row["label"] / row["name"]) as image:
                        pixel_rows.append(deterministic_transform(image.convert("RGB")))
                    prompts.append(f"a {str(row['label']).replace('_', ' ').lower()} painting")
                pixels = torch.stack(pixel_rows).to(device, dtype=torch.float16)
                tokens = tokenizer(
                    prompts,
                    max_length=tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(device)
                with torch.no_grad():
                    latents = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor
                    hidden = text_encoder(tokens)[0]
                gradient_sums = [
                    torch.zeros((len(positions),) + tuple(parameter.shape), device=device, dtype=torch.float32)
                    for parameter in active
                ]
                # The LoRA Linear hook computes all per-example gradients from
                # one backward per timestep chunk instead of one backward per
                # datapoint. This is the later ArtBench-2 fast path.
                timestep_chunk = 5
                for chunk_start in range(0, len(timesteps), timestep_chunk):
                    chunk = timesteps[chunk_start : chunk_start + timestep_chunk]
                    noises = []
                    for position in positions:
                        rows = []
                        for timestep in chunk:
                            generator = torch.Generator(device=device)
                            generator.manual_seed(
                                artbench_seed(
                                    "endpoint_tracin_train",
                                    checkpoint_index,
                                    int(position),
                                    timestep,
                                )
                            )
                            rows.append(
                                torch.randn(
                                    latents[0:1].shape,
                                    generator=generator,
                                    device=device,
                                    dtype=latents.dtype,
                                )[0]
                            )
                        noises.append(torch.stack(rows))
                    noise = torch.stack(noises)
                    count = len(chunk)
                    flat_latents = latents[:, None].expand(
                        len(positions), count, *latents.shape[1:]
                    ).reshape(-1, *latents.shape[1:])
                    flat_hidden = hidden[:, None].expand(
                        len(positions), count, *hidden.shape[1:]
                    ).reshape(-1, *hidden.shape[1:])
                    flat_noise = noise.reshape(-1, *latents.shape[1:])
                    flat_t = torch.as_tensor(chunk, device=device, dtype=torch.long).repeat(
                        len(positions)
                    )
                    capture.start(len(flat_latents))
                    prediction = unet(
                        scheduler.add_noise(flat_latents, flat_noise, flat_t),
                        flat_t,
                        flat_hidden,
                    ).sample
                    losses = functional.mse_loss(
                        prediction.float(), flat_noise.float(), reduction="none"
                    ).mean(dim=(1, 2, 3))
                    chunk_gradients = capture.gradients(losses)
                    for total, values in zip(gradient_sums, chunk_gradients):
                        total.add_(
                            values.reshape(
                                len(positions), count, *values.shape[1:]
                            ).sum(dim=1),
                            alpha=1.0 / len(timesteps),
                        )
                raw_features = projector.project_batch(gradient_sums)
                query = query_features
                if args.method == "gas":
                    query = query / query.norm(dim=1, keepdim=True).clamp_min(1e-12)
                    raw_features = raw_features / raw_features.norm(dim=1, keepdim=True).clamp_min(1e-12)
                raw_scores[:, start:end] += checkpoint_lr * (query @ raw_features.T).double().cpu().numpy()
                if not args.no_adamw:
                    directions, reconstructed_lr = adamw_full_direction_batch(
                        active, gradient_sums, optimizer, max_grad_norm=1.0
                    )
                    if not np.isclose(checkpoint_lr, reconstructed_lr):
                        raise RuntimeError("checkpoint optimizer LR mismatch")
                    adamw_features = projector.project_batch(directions)
                    if args.method == "gas":
                        adamw_features = adamw_features / adamw_features.norm(dim=1, keepdim=True).clamp_min(1e-12)
                    assert adamw_scores is not None
                    adamw_scores[:, start:end] += checkpoint_lr * (query @ adamw_features.T).double().cpu().numpy()
                progress.update(1)
            capture.close()
    progress.close()
    metadata = {
        "format_version": 2,
        "experiment": "artbench2",
        "method": args.method,
        "tag": tag,
        "normalization": manifest["query_reduction"],
        "score_normalization": manifest["score_normalization"],
        "projection": "countsketch",
        "projection_dim": args.projection_dim,
        "timesteps": spec.timestep_count,
        "checkpoint_pairs": len(parts),
        "query_manifest_sha256": hashlib.sha256(args.query_manifest.read_bytes()).hexdigest(),
        "scope": "all",
        "scope_size": len(all_positions),
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
