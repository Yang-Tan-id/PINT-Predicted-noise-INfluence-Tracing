from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from pint.sampling.queries import artbench2_queries
from pint.training.model_versions import ARTBENCH_MODEL_ID, ARTBENCH_MODEL_REVISION


def _model_seed(model_dir: Path) -> int:
    config_path = model_dir / "training_config.json"
    if not config_path.is_file():
        raise RuntimeError(
            f"Missing ArtBench-2 model seed metadata: {config_path}; "
            "retrain with the reproducible entrypoint"
        )
    return int(json.loads(config_path.read_text(encoding="utf-8"))["seed"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sample ArtBench-2 queries")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-queries", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=ARTBENCH_MODEL_ID)
    parser.add_argument("--model-revision", default=ARTBENCH_MODEL_REVISION)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--max-replacement-attempts", type=int, default=100)
    args = parser.parse_args(argv)
    if args.num_queries <= 0:
        parser.error("--num-queries must be positive")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.max_replacement_attempts <= 0:
        parser.error("--max-replacement-attempts must be positive")
    lora_dir = args.model_dir / "final"
    if not lora_dir.is_dir():
        raise FileNotFoundError(f"LoRA weights not found: {lora_dir}; run 01_train_model.py first")
    model_seed = _model_seed(args.model_dir)

    try:
        import torch
        from diffusers import DDIMScheduler, StableDiffusionPipeline
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise SystemExit("ArtBench-2 sampling requires the train-artbench dependencies") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("ArtBench-2 sampling requires an NVIDIA GPU")

    pretrained = {"revision": args.model_revision}
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16, **pretrained
    )
    training_timesteps = int(pipeline.scheduler.config.num_train_timesteps)
    if training_timesteps % args.steps:
        raise ValueError(
            f"--steps must divide the model's {training_timesteps} training timesteps"
        )
    timestep_stride = training_timesteps // args.steps
    pipeline.scheduler = DDIMScheduler.from_config(
        pipeline.scheduler.config,
        steps_offset=timestep_stride - 1,
    )
    pipeline.unet.load_attn_procs(lora_dir)
    pipeline = pipeline.to("cuda:0")
    pipeline.set_progress_bar_config(disable=True)

    records = artbench2_queries(args.num_queries, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    next_replacement_seed = max(int(record["initial_seed"]) for record in records) + 1
    for record in tqdm(records, desc="Sampling ArtBench-2 queries", unit="query"):
        requested_seed = int(record["initial_seed"])
        actual_seed = requested_seed
        replacement_attempts = 0
        while True:
            generator = torch.Generator(device=pipeline.device).manual_seed(actual_seed)
            final_latent = None

            def capture_final_latent(_step, _timestep, latents):
                nonlocal final_latent
                final_latent = latents.detach().float().cpu().numpy()

            output = pipeline(
                str(record["prompt"]),
                height=256,
                width=256,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                callback=capture_final_latent,
                callback_steps=1,
            )
            image = output.images[0]
            detected = getattr(output, "nsfw_content_detected", None)
            safety_flagged = (
                detected is not None and len(detected) > 0 and bool(detected[0])
            )
            all_black = all(
                low == 0 and high == 0 for low, high in image.convert("RGB").getextrema()
            )
            if not safety_flagged and not all_black:
                break
            replacement_attempts += 1
            if replacement_attempts > args.max_replacement_attempts:
                raise RuntimeError(
                    "Exhausted ArtBench-2 replacement seeds for "
                    f"prompt={record['prompt']!r}, requested_seed={requested_seed}"
                )
            actual_seed = next_replacement_seed
            next_replacement_seed += 1
        expected_timesteps = [
            training_timesteps - 1 - timestep_stride * index
            for index in range(args.steps)
        ]
        actual_timesteps = [int(value) for value in pipeline.scheduler.timesteps.tolist()]
        if actual_timesteps != expected_timesteps:
            raise RuntimeError(
                "ArtBench-2 DDIM schedule mismatch: "
                f"actual={actual_timesteps}, expected={expected_timesteps}"
            )
        filename = f"q{int(record['query_id']):02d}.png"
        latent_filename = f"q{int(record['query_id']):02d}_final_latent.npy"
        if final_latent is None:
            raise RuntimeError("ArtBench-2 pipeline did not return a final latent")
        image.save(args.output_dir / filename)
        np.save(args.output_dir / latent_filename, final_latent)
        record["requested_seed"] = requested_seed
        record["initial_seed"] = actual_seed
        record["replacement_attempts"] = replacement_attempts
        record["model_seed"] = model_seed
        record["file"] = filename
        record["latent_file"] = latent_filename
        record["sampler"] = "ddim_eta0"
        record["num_inference_steps"] = args.steps

    (args.output_dir / "manifest.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(records)} queries to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
