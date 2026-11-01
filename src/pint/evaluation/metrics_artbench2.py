from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _trajectory_metrics(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"trajectory shape mismatch: {left.shape} vs {right.shape}")
    return float(np.mean((left[-1] - right[-1]) ** 2)), float(np.mean((left - right) ** 2))


def _expected_timesteps(training_timesteps: int, steps: int) -> list[int]:
    if steps < 1 or training_timesteps % steps:
        raise ValueError(
            f"training timesteps {training_timesteps} must be divisible by steps {steps}"
        )
    stride = training_timesteps // steps
    return [training_timesteps - 1 - stride * index for index in range(steps)]


def _is_all_black(image) -> bool:
    return all(low == 0 and high == 0 for low, high in image.convert("RGB").getextrema())


def evaluate(
    model_dir: Path,
    reference_dir: Path,
    query_manifest: Path,
    query_id: int,
    output: Path,
) -> None:
    import torch
    from diffusers import DDIMScheduler, StableDiffusionPipeline

    from pint.training.model_versions import ARTBENCH_MODEL_ID, ARTBENCH_MODEL_REVISION

    if not torch.cuda.is_available():
        raise RuntimeError("ArtBench-2 evaluation requires an NVIDIA GPU")
    device = torch.device("cuda:0")
    reference_config = json.loads(
        (reference_dir / "training_config.json").read_text(encoding="utf-8")
    )
    model_id = str(reference_config.get("model", ARTBENCH_MODEL_ID))
    revision = str(reference_config.get("model_revision", ARTBENCH_MODEL_REVISION))
    records = json.loads(query_manifest.read_text(encoding="utf-8"))
    record = next(item for item in records if int(item["query_id"]) == query_id)
    steps = int(record.get("num_inference_steps", 50))
    seed = int(record["initial_seed"])

    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    training_timesteps = int(pipeline.scheduler.config.num_train_timesteps)
    expected_timesteps = _expected_timesteps(training_timesteps, steps)
    pipeline.scheduler = DDIMScheduler.from_config(
        pipeline.scheduler.config,
        steps_offset=training_timesteps // steps - 1,
    )
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=False)

    def sample(lora_dir: Path) -> tuple[object, np.ndarray, np.ndarray]:
        pipeline.unet.load_attn_procs(lora_dir)
        pipeline.unet.eval()
        states: list[np.ndarray] = []

        def capture(_step: int, _timestep: int, latents: torch.Tensor) -> None:
            states.append(latents[0].detach().float().cpu().numpy().copy())

        generator = torch.Generator(device=device).manual_seed(seed)
        result = pipeline(
            str(record["prompt"]),
            height=256,
            width=256,
            num_inference_steps=steps,
            guidance_scale=7.5,
            generator=generator,
            callback=capture,
            callback_steps=1,
        )
        actual_timesteps = [int(value) for value in pipeline.scheduler.timesteps.tolist()]
        if actual_timesteps != expected_timesteps:
            raise RuntimeError(
                "ArtBench-2 DDIM schedule mismatch: "
                f"actual={actual_timesteps}, expected={expected_timesteps}"
            )
        if len(states) != steps:
            raise RuntimeError(f"expected {steps} trajectory states, found {len(states)}")
        image = result.images[0].convert("RGB")
        if _is_all_black(image):
            raise RuntimeError("ArtBench-2 generation produced an all-black image")
        array = np.asarray(image, dtype=np.float32) / 255.0
        return image, np.stack(states), array

    reference_image, reference_trajectory, reference_pixels = sample(reference_dir / "final")
    candidate_image, candidate_trajectory, candidate_pixels = sample(model_dir / "final")
    endpoint_mse = float(np.mean((reference_pixels - candidate_pixels) ** 2))
    _latent_endpoint_mse, trajectory_mse = _trajectory_metrics(
        reference_trajectory, candidate_trajectory
    )

    assets = model_dir / "evaluation_assets"
    assets.mkdir(parents=True, exist_ok=True)
    reference_image.save(assets / "reference.png")
    candidate_image.save(assets / "candidate.png")
    np.savez_compressed(
        assets / "trajectories.npz",
        reference=reference_trajectory.astype(np.float16),
        candidate=candidate_trajectory.astype(np.float16),
        timesteps=np.asarray(pipeline.scheduler.timesteps.cpu(), dtype=np.int64),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        query_ids=np.asarray([query_id], dtype=np.int32),
        endpoint_mse=np.asarray([endpoint_mse], dtype=np.float64),
        trajectory_mse=np.asarray([trajectory_mse], dtype=np.float64),
        pixel_l2=np.asarray([np.sqrt(endpoint_mse)], dtype=np.float64),
        clip_distance=np.asarray([np.nan], dtype=np.float64),
        dino_distance=np.asarray([np.nan], dtype=np.float64),
        lpips=np.asarray([np.nan], dtype=np.float64),
        trajectory_reduction=np.asarray("latent_state_mse"),
    )
