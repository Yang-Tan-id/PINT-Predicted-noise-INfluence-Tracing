from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from pint.datasets.paths import data_root
from pint.training.backends import x3_model
from pint.training.backends.x3_dataset import ColorGridDataset


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _learning_rate(step: int, total_steps: int, peak: float, warmup: float) -> float:
    warmup_steps = int(math.ceil(total_steps * warmup))
    if warmup_steps and step < warmup_steps:
        return peak * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * peak * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def _update_ema(ema_model, model, decay: float) -> None:
    for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
        ema_parameter.mul_(decay).add_(parameter, alpha=1.0 - decay)


def _train_family(args: argparse.Namespace, family: str) -> None:
    _set_seed(args.seed)
    dataset = ColorGridDataset(str(args.dataset), grid_size=3)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=args.device.startswith("cuda"),
    )
    image, condition = dataset[0]
    model = x3_model.CondEpsModel(
        in_ch=image.shape[0],
        cond_dim=condition.numel(),
        base_ch=args.base_channels,
        time_dim=args.time_embedding_dim,
    ).to(args.device)
    ema_model = copy.deepcopy(model).to(args.device).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)

    schedule = x3_model.make_linear_schedule(args.timesteps, device=args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * len(loader)
    checkpoint_every = max(1, args.epochs // args.num_checkpoints)
    # Keep the original experiment layout because downstream attribution scripts
    # resolve baseline checkpoints as MODEL_DIR/base/<family>/epoch_XXXX.pt.
    output_dir = args.output_dir / "base" / family
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = vars(args).copy()
    manifest.update(
        family=family,
        dataset=str(args.dataset.resolve()),
        output_dir=str(output_dir.resolve()),
        vocabulary=dataset.vocab,
    )
    for key, value in list(manifest.items()):
        if isinstance(value, Path):
            manifest[key] = str(value)
    (output_dir / "training_config.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    metadata = {
        "kind": "base",
        "family": family,
        "seed": int(args.seed),
        "gpu": int(visible_devices.split(",")[0]) if visible_devices else None,
        "base_csv": str(args.dataset),
        "mask_path": None,
        "lds_seed": None,
        "subset_id": None,
        "n_train": len(dataset),
    }
    (output_dir / "meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        progress = tqdm(loader, desc=f"{family} epoch {epoch}/{args.epochs}", dynamic_ncols=True)
        running_loss = 0.0
        for batch_index, (clean, condition) in enumerate(progress, start=1):
            clean = clean.to(args.device, non_blocking=True)
            condition = condition.to(args.device, non_blocking=True)
            if family == "unprompted":
                condition = torch.zeros_like(condition)
            rate = _learning_rate(
                global_step, total_steps, args.learning_rate, args.warmup_ratio
            )
            for group in optimizer.param_groups:
                group["lr"] = rate
            timestep = torch.randint(
                0, args.timesteps, (clean.shape[0],), device=args.device
            )
            noise = torch.randn_like(clean)
            noisy = x3_model.q_sample(clean, timestep, noise, schedule)
            loss = F.mse_loss(model(noisy, timestep, condition), noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            _update_ema(ema_model, model, args.ema_decay)
            global_step += 1
            running_loss += loss.detach().item()
            progress.set_postfix(loss=f"{running_loss / batch_index:.5f}", lr=f"{rate:.2e}")

        if epoch % checkpoint_every == 0 or epoch == args.epochs:
            checkpoint_config = {
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "lr_schedule": "cosine_warmup",
                "lr_warmup_ratio": args.warmup_ratio,
                "weight_decay": args.weight_decay,
                "adam_b1": 0.9,
                "adam_b2": 0.999,
                "adam_eps": 1e-8,
                "grad_clip_norm": args.grad_clip_norm,
                "ema_decay": args.ema_decay,
                "timesteps": args.timesteps,
                "base_channels": args.base_channels,
                "time_emb_dim": args.time_embedding_dim,
            }
            checkpoint = {
                "model_state": model.state_dict(),
                "ema_model_state": ema_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "eta": float(rate),
                "learning_rate_at_checkpoint": float(rate),
                "T": args.timesteps,
                "cond_dim": len(dataset.vocab),
                "vocab": dataset.vocab,
                "grid_size": 3,
                "base_ch": args.base_channels,
                "time_dim": args.time_embedding_dim,
                "family": family,
                "unprompted": family == "unprompted",
                "train_indices": None,
                "config": checkpoint_config,
            }
            torch.save(checkpoint, output_dir / f"epoch_{epoch:04d}.pt")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the x3 pixel DDPM")
    parser.add_argument("--dataset", type=Path, default=data_root() / "x3" / "67_50000.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--family", choices=("prompted", "unprompted", "both"), default="both")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--time-embedding-dim", type=int, default=128)
    parser.add_argument("--num-checkpoints", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args(argv)
    if not args.dataset.is_file():
        parser.error(f"dataset not found: {args.dataset}; run experiments/x3/00_prepare_dataset.py first")
    if args.num_checkpoints < 1 or args.num_checkpoints > args.epochs:
        parser.error("--num-checkpoints must be between 1 and --epochs")
    if args.epochs % args.num_checkpoints:
        parser.error("--epochs must be divisible by --num-checkpoints")
    families = ("prompted", "unprompted") if args.family == "both" else (args.family,)
    for family in families:
        _train_family(args, family)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
