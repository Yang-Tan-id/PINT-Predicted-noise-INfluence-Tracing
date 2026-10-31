from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from pint.baselines.artbench_projection import ArtBenchProjector, artbench_seed
from pint.baselines.common import (
    atomic_savez,
    baseline_spec,
    query_part_path,
    run_tag,
    write_query_manifest,
)
from pint.attribution.artbench2 import LinearGradSampleCapture
from pint.query_gradients.artbench2 import _build_unet, _checkpoints, _load_lora_checkpoint
from pint.query_gradients.common import checkpoint_pair_range, read_query_manifest, timestep_grid
from pint.training.model_versions import ARTBENCH_MODEL_ID, ARTBENCH_MODEL_REVISION


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("artbench2",), required=True)
    parser.add_argument("--method", choices=("gas", "tracincp", "retrac"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--projection-dim", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--model", default=ARTBENCH_MODEL_ID)
    parser.add_argument("--model-revision", default=ARTBENCH_MODEL_REVISION)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        import torch
        import torch.nn.functional as functional
        from diffusers import DDPMScheduler
        from tqdm.auto import tqdm
        from transformers import CLIPTextModel, CLIPTokenizer
    except ImportError as exc:
        raise SystemExit("ArtBench-2 baselines require train-artbench") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("ArtBench-2 baselines require an NVIDIA GPU")
    device = torch.device("cuda:0")
    config = json.loads((args.model_dir / "training_config.json").read_text(encoding="utf-8"))
    model_seed = int(config["seed"])
    rank = int(config.get("rank", 128))
    if args.model == ARTBENCH_MODEL_ID:
        args.model = config.get("model", args.model)
    if args.model_revision == ARTBENCH_MODEL_REVISION:
        args.model_revision = config.get("model_revision", args.model_revision)
    checkpoints = _checkpoints(args.model_dir)
    source_count = len(checkpoints) - 1 if args.method == "retrac" else len(checkpoints)
    owned = checkpoint_pair_range(source_count, args.shard_index, args.shard_count)
    if not owned:
        return 0
    _, queries = read_query_manifest(args.query_manifest)
    spec = baseline_spec(args.method, "artbench2")
    tag = run_tag(spec, args.projection_dim)
    write_query_manifest(
        args.output_dir,
        spec=spec,
        tag=tag,
        projection_dim=args.projection_dim,
        query_manifest=args.query_manifest,
        checkpoints=checkpoints,
        source_checkpoint_count=source_count,
    )
    tokenizer = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer", revision=args.model_revision)
    text_encoder = CLIPTextModel.from_pretrained(
        args.model, subfolder="text_encoder", revision=args.model_revision, torch_dtype=torch.float16
    ).to(device).eval()
    scheduler = DDPMScheduler.from_pretrained(args.model, subfolder="scheduler", revision=args.model_revision)
    timesteps = timestep_grid("artbench2", int(scheduler.config.num_train_timesteps))
    latent_bank, hidden_bank, query_ids = [], [], []
    with torch.no_grad():
        for start in range(0, len(queries), args.batch_size):
            batch = queries[start : start + args.batch_size]
            latents = []
            for query in batch:
                latent_name = query.get("latent_file")
                if not latent_name:
                    raise RuntimeError("ArtBench-2 query latents are missing; rerun step 02")
                latents.append(np.load(args.query_manifest.parent / str(latent_name))[0])
            tokens = tokenizer(
                [str(query["prompt"]) for query in batch],
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            hidden = text_encoder(tokens)[0]
            latent_bank.extend(torch.from_numpy(np.stack(latents)).half())
            hidden_bank.extend(hidden.cpu())
            query_ids.extend(int(query["query_id"]) for query in batch)
    unet, layers = _build_unet(args.model, args.model_revision, rank, device)
    progress = tqdm(
        total=len(owned) * len(queries),
        desc=f"ArtBench-2 {args.method} query shard {args.shard_index + 1}/{args.shard_count}",
        unit="query",
        dynamic_ncols=True,
    )
    for checkpoint_index in owned:
        output_path = query_part_path(args.output_dir, checkpoint_index, tag)
        if output_path.is_file():
            progress.update(len(queries))
            continue
        layers = _load_lora_checkpoint(unet, layers, checkpoints[checkpoint_index])
        active = [parameter for parameter in layers.parameters() if parameter.requires_grad]
        capture = LinearGradSampleCapture(unet, active) if args.method == "retrac" else None
        projector = ArtBenchProjector(
            args.projection_dim, artbench_seed("sd_artbench", "projection", checkpoint_index)
        )
        query_features = []
        for query_id, latent_cpu, hidden_cpu in zip(query_ids, latent_bank, hidden_bank):
            latent = latent_cpu.unsqueeze(0).to(device)
            hidden = hidden_cpu.unsqueeze(0).to(device)
            total = torch.zeros(args.projection_dim, device=device, dtype=torch.float32)
            gradient_total = [torch.zeros_like(parameter, dtype=torch.float32) for parameter in active]
            timestep_chunk = 5
            for chunk_start in range(0, len(timesteps), timestep_chunk):
                chunk = timesteps[chunk_start : chunk_start + timestep_chunk]
                noise_rows = []
                for timestep in chunk:
                    generator = torch.Generator(device=device)
                    generator.manual_seed(
                        artbench_seed(
                            "retrac_query" if args.method == "retrac" else "endpoint_tracin",
                            checkpoint_index,
                            query_id,
                            int(timestep),
                        )
                    )
                    noise_rows.append(
                        torch.randn(
                            latent.shape,
                            generator=generator,
                            device=device,
                            dtype=latent.dtype,
                        )[0]
                    )
                noise = torch.stack(noise_rows)
                count = len(chunk)
                latent_batch = latent.expand(count, *latent.shape[1:])
                hidden_batch = hidden.expand(count, *hidden.shape[1:])
                t = torch.as_tensor(chunk, device=device, dtype=torch.long)
                if capture is not None:
                    capture.start(count)
                prediction = unet(
                    scheduler.add_noise(latent_batch, noise, t), t, hidden_batch
                ).sample
                losses = functional.mse_loss(
                    prediction.float(), noise.float(), reduction="none"
                ).mean(dim=(1, 2, 3))
                if capture is not None:
                    gradients = capture.gradients(losses)
                    total.add_(
                        projector.project_batch(
                            gradients, full_l2_normalize=True
                        ).sum(dim=0),
                        alpha=1.0 / len(timesteps),
                    )
                else:
                    gradients = torch.autograd.grad(
                        losses.mean(), active, allow_unused=True
                    )
                    gradients = [
                        torch.zeros_like(parameter) if value is None else value
                        for parameter, value in zip(active, gradients)
                    ]
                    for destination, value in zip(gradient_total, gradients):
                        destination.add_(value.float(), alpha=count / len(timesteps))
            if capture is None:
                total.copy_(projector.project(gradient_total))
            query_features.append(total.cpu().numpy())
            progress.update(1)
        if capture is not None:
            capture.close()
        atomic_savez(
            output_path,
            query_features=np.stack(query_features).astype(np.float32),
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
