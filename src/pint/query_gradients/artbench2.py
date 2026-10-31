from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Sequence

import numpy as np

from pint.query_gradients.common import (
    atomic_savez,
    checkpoint_pair_range,
    part_path,
    read_query_manifest,
    stable_torch_seed,
    timestep_grid,
    write_run_manifest,
)
from pint.query_gradients.torch_projector import CountSketchProjector
from pint.training.model_versions import ARTBENCH_MODEL_ID, ARTBENCH_MODEL_REVISION


CHECKPOINT_PATTERN = re.compile(r"checkpoint-epoch-(?P<epoch>\d+)$")


def _checkpoints(model_dir: Path) -> list[Path]:
    found = []
    for path in model_dir.glob("checkpoint-epoch-*"):
        match = CHECKPOINT_PATTERN.match(path.name)
        if match and path.is_dir():
            found.append((int(match.group("epoch")), path))
    if len(found) < 2:
        raise FileNotFoundError(f"Need at least two ArtBench-2 checkpoints under {model_dir}")
    return [path for _, path in sorted(found)]


def _load_lora_checkpoint(unet, layers, checkpoint: Path):
    """Load the explicit snapshot, with compatibility for older Accelerator states."""
    lora_dir = checkpoint / "lora"
    if lora_dir.is_dir():
        from diffusers.loaders import AttnProcsLayers

        unet.load_attn_procs(lora_dir)
        return AttnProcsLayers(unet.attn_processors)

    import torch

    candidates = sorted(checkpoint.glob("pytorch_model*.bin"))
    candidates.extend(sorted(checkpoint.glob("model*.safetensors")))
    if not candidates:
        raise FileNotFoundError(
            f"No LoRA snapshot or Accelerator model state found in {checkpoint}"
        )
    path = candidates[0]
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu")
    missing, unexpected = layers.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Incompatible legacy LoRA state {path}: missing={missing}, unexpected={unexpected}"
        )
    return layers


def _build_unet(model_id: str, revision: str, rank: int, device):
    import torch
    from diffusers import UNet2DConditionModel

    from pint.training.backends.artbench_lora import create_lora_layers

    unet = UNet2DConditionModel.from_pretrained(
        model_id,
        subfolder="unet",
        revision=revision,
        torch_dtype=torch.float16,
    ).to(device)
    unet.requires_grad_(False)
    layers = create_lora_layers(unet, rank)
    unet.eval()
    return unet, layers


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute checkpoint-major ArtBench-2 PINT gradients")
    parser.add_argument("--experiment", choices=("artbench2",), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--projection-dim", type=int, default=4096)
    parser.add_argument("--model", default=ARTBENCH_MODEL_ID)
    parser.add_argument("--model-revision", default=ARTBENCH_MODEL_REVISION)
    parser.add_argument("--shard-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--shard-count", type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.projection_dim <= 0:
        parser.error("--batch-size and --projection-dim must be positive")

    try:
        import torch
        from diffusers import DDPMScheduler
        from transformers import CLIPTextModel, CLIPTokenizer
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise SystemExit("ArtBench-2 query gradients require train-artbench") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("ArtBench-2 query gradients require an NVIDIA GPU")
    device = torch.device("cuda:0")

    training_config_path = args.model_dir / "training_config.json"
    if not training_config_path.is_file():
        raise FileNotFoundError(training_config_path)
    training_config = json.loads(training_config_path.read_text(encoding="utf-8"))
    model_seed = int(training_config["seed"])
    rank = int(training_config.get("rank", 128))
    configured_model = training_config.get("model", args.model)
    configured_revision = training_config.get("model_revision", args.model_revision)
    if args.model == ARTBENCH_MODEL_ID:
        args.model = configured_model
    if args.model_revision == ARTBENCH_MODEL_REVISION:
        args.model_revision = configured_revision

    checkpoints = _checkpoints(args.model_dir)
    try:
        pair_indices = checkpoint_pair_range(
            len(checkpoints) - 1, args.shard_index, args.shard_count
        )
    except ValueError as exc:
        parser.error(str(exc))
    if not pair_indices:
        return 0
    _, queries = read_query_manifest(args.query_manifest)
    if any(int(query.get("model_seed", -1)) != model_seed for query in queries):
        raise RuntimeError(
            "ArtBench-2 query bank and training checkpoints must use the same model seed"
        )
    tokenizer = CLIPTokenizer.from_pretrained(
        args.model, subfolder="tokenizer", revision=args.model_revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.model,
        subfolder="text_encoder",
        revision=args.model_revision,
        torch_dtype=torch.float16,
    ).to(device).eval()
    scheduler = DDPMScheduler.from_pretrained(
        args.model, subfolder="scheduler", revision=args.model_revision
    )
    timesteps = timestep_grid(
        "artbench2", int(scheduler.config.num_train_timesteps)
    )
    if any(
        int(query.get("num_inference_steps", -1)) != len(timesteps)
        for query in queries
    ):
        raise RuntimeError(
            "ArtBench-2 query sampling timesteps do not match query-gradient "
            "timesteps; rerun 02_sample_queries.py with the default settings"
        )
    write_run_manifest(
        args.output_dir,
        experiment="artbench2",
        query_manifest=args.query_manifest,
        checkpoints=checkpoints,
        timesteps=timesteps,
        projection_dim=args.projection_dim,
        batch_size=args.batch_size,
    )
    current_unet, current_layers = _build_unet(
        args.model, args.model_revision, rank, device
    )
    target_unet, target_layers = _build_unet(
        args.model, args.model_revision, rank, device
    )
    current_layers = _load_lora_checkpoint(
        current_unet, current_layers, checkpoints[pair_indices.start]
    )

    # Read the exact final latents captured by 02_sample_queries.py. Re-encoding
    # the PNG would introduce a decode/quantize/encode round trip.
    latent_bank = []
    hidden_bank = []
    query_ids = []
    with torch.no_grad():
        for start in range(0, len(queries), args.batch_size):
            batch = queries[start : start + args.batch_size]
            latents = []
            for query in batch:
                latent_name = query.get("latent_file")
                if not latent_name:
                    raise RuntimeError(
                        "ArtBench-2 query manifest has no final latents; rerun "
                        "02_sample_queries.py with the current code"
                    )
                latent_path = args.query_manifest.parent / str(latent_name)
                latent = np.load(latent_path).astype(np.float32)
                if latent.ndim != 4 or latent.shape[0] != 1:
                    raise ValueError(f"Invalid ArtBench-2 latent shape {latent.shape}: {latent_path}")
                latents.append(latent[0])
            tokens = tokenizer(
                [str(query["prompt"]) for query in batch],
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            hidden = text_encoder(tokens)[0]
            latent_bank.extend(torch.from_numpy(np.stack(latents)).half())
            hidden_bank.extend(hidden.detach().cpu())
            query_ids.extend(int(query["query_id"]) for query in batch)

    progress = tqdm(
        total=(
            len(pair_indices)
            * math.ceil(len(queries) / args.batch_size)
            * len(timesteps)
        ),
        desc=f"ArtBench-2 query gradients shard {args.shard_index + 1}/{args.shard_count}",
        unit="batch-timestep",
        dynamic_ncols=True,
    )
    for checkpoint_index in pair_indices:
        target_layers = _load_lora_checkpoint(
            target_unet, target_layers, checkpoints[checkpoint_index + 1]
        )
        output_path = part_path(args.output_dir, checkpoint_index)
        if output_path.is_file():
            current_unet, target_unet = target_unet, current_unet
            current_layers, target_layers = target_layers, current_layers
            progress.update(math.ceil(len(queries) / args.batch_size) * len(timesteps))
            continue
        active = [parameter for parameter in current_layers.parameters() if parameter.requires_grad]
        projector = CountSketchProjector(
            active,
            args.projection_dim,
            model_seed,
            "traj_tracin_projection",
            checkpoint_index,
        )
        noise_bank = []
        generator = torch.Generator(device=device)
        for timestep in timesteps:
            generator.manual_seed(
                stable_torch_seed(
                    model_seed, checkpoint_index, int(timestep), "query_noise"
                )
            )
            noise_bank.append(
                torch.randn(
                    (1, *latent_bank[0].shape),
                    generator=generator,
                    device=device,
                    dtype=torch.float16,
                )
            )
        checkpoint_features = []
        checkpoint_gradient_norms = []
        checkpoint_norms = []
        for start in range(0, len(queries), args.batch_size):
            end = min(start + args.batch_size, len(queries))
            clean = torch.stack(latent_bank[start:end]).to(device)
            hidden = torch.stack(hidden_bank[start:end]).to(device)
            batch_features = []
            batch_norms = []
            for timestep, shared_noise in zip(timesteps, noise_bank):
                t = torch.full(
                    (clean.shape[0],), int(timestep), device=device, dtype=torch.long
                )
                noisy = scheduler.add_noise(clean, shared_noise.expand_as(clean), t)
                prediction = current_unet(noisy, t, hidden).sample
                with torch.no_grad():
                    target = target_unet(noisy, t, hidden).sample
                    delta = target - prediction.detach()
                timestep_features = []
                timestep_norms = []
                for local_index in range(clean.shape[0]):
                    delta_norm = delta[local_index].float().norm()
                    direction = delta[local_index].float() / delta_norm.clamp_min(1e-12)
                    scalar = (prediction[local_index].float() * direction).mean()
                    gradients = torch.autograd.grad(
                        scalar,
                        active,
                        retain_graph=local_index + 1 < clean.shape[0],
                        allow_unused=False,
                    )
                    timestep_features.append(projector(gradients).detach().cpu().numpy())
                    timestep_norms.append(delta_norm.item())
                batch_features.append(np.stack(timestep_features))
                batch_norms.append(np.asarray(timestep_norms, dtype=np.float32))
                progress.update(1)
            features = np.stack(batch_features, axis=1)
            checkpoint_gradient_norms.append(
                np.linalg.norm(features, axis=-1).astype(np.float32)
            )
            checkpoint_features.append(features.astype(np.float32))
            checkpoint_norms.append(np.stack(batch_norms, axis=1))
            print(
                f"checkpoint {checkpoint_index + 1}/{len(checkpoints) - 1}: "
                f"queries {start + 1}-{end}/{len(queries)}",
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
        current_unet, target_unet = target_unet, current_unet
        current_layers, target_layers = target_layers, current_layers
    progress.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
