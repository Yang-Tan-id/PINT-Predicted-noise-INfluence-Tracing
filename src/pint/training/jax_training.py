from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from pint.datasets.paths import data_root
from pint.training.backends.jax_pixel import TrainConfig, train


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a PINT JAX pixel DDPM")
    parser.add_argument("--experiment", required=True, choices=("shape3d", "celeba"))
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("gpu", "cpu", "auto"), default="gpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-checkpoints", type=int, default=50)
    parser.add_argument("--num-devices", type=int)
    parser.add_argument("--no-bfloat16", action="store_true")
    parser.add_argument("--resume-from")
    args = parser.parse_args(argv)

    if args.dataset is None:
        size = "20000" if args.experiment == "shape3d" else "10000"
        args.dataset = data_root() / args.experiment / size / "dataset.npz"
    if not args.dataset.is_file():
        parser.error(
            f"dataset not found: {args.dataset}; run "
            f"experiments/{args.experiment}/00_prepare_dataset.py first"
        )
    if args.num_checkpoints < 1 or args.num_checkpoints > args.epochs:
        parser.error("--num-checkpoints must be between 1 and --epochs")
    if args.epochs % args.num_checkpoints:
        parser.error("--epochs must be divisible by --num-checkpoints")

    unconditional = args.experiment == "celeba"
    config = TrainConfig(
        data_root=str(args.dataset),
        batch_names=None,
        class_names=None,
        use_test=False,
        model_type="unet",
        image_size=64,
        in_channels=3,
        base_channels=160,
        channel_mults=(1, 2, 2),
        num_res_blocks=2,
        time_emb_dim=128,
        num_classes=1 if unconditional else 34,
        class_cond=not unconditional,
        cond_mode="multi_hot",
        dropout=0.1,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_schedule="cosine_warmup",
        lr_warmup_ratio=0.1,
        weight_decay=1e-4,
        adam_b1=0.9,
        adam_b2=0.999,
        adam_eps=1e-8,
        grad_clip_norm=1.0,
        ema_decay=0.999,
        log_every=100,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        predict_x0=False,
        prefer_device=args.device,
        use_bfloat16=not args.no_bfloat16,
        use_data_parallel=True,
        num_devices=args.num_devices,
        checkpoint_dir=str(args.output_dir),
        save_every_epochs=args.epochs // args.num_checkpoints,
        keep_last_k=None,
        resume_from=args.resume_from,
        num_workers=0,
        use_tqdm=True,
        prefetch_size=4,
        use_wandb=False,
        wandb_run_name=f"pint-{args.experiment}-seed{args.seed}",
    )
    train(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
