import argparse
import json
import math
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.optimization import get_scheduler
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from pint.config import repo_root
from pint.datasets.paths import data_root
from pint.training.model_versions import ARTBENCH_MODEL_ID, ARTBENCH_MODEL_REVISION


def label_to_prompt(label: str) -> str:
    return f"a {label.replace('_', ' ').lower()} painting"


def default_output_root() -> Path:
    import os

    value = os.environ.get("PINT_OUTPUT_ROOT")
    if value:
        return Path(value).expanduser().resolve()
    return repo_root() / "outputs"


class IndexedArtBenchDataset(Dataset):
    def __init__(
        self,
        metadata_csv: Path,
        image_root: Path,
        index_path: Path,
        tokenizer: CLIPTokenizer,
        resolution: int,
    ) -> None:
        frame = pd.read_csv(metadata_csv)
        with index_path.open("rb") as handle:
            indices = pickle.load(handle)
        self.frame = frame.loc[indices].copy()
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, item: int):
        row = self.frame.iloc[item]
        image_path = self.image_root / row["label"] / row["name"]
        with Image.open(image_path) as image:
            pixel_values = self.transform(image.convert("RGB"))
        tokenized = self.tokenizer(
            label_to_prompt(row["label"]),
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": tokenized.input_ids[0],
        }


def create_lora_layers(unet: UNet2DConditionModel, rank: int) -> AttnProcsLayers:
    processors = {}
    reversed_channels = list(reversed(unet.config.block_out_channels))
    for name in unet.attn_processors.keys():
        cross_attention_dim = (
            None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name.split(".")[1])
            hidden_size = reversed_channels[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name.split(".")[1])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            raise ValueError(f"Unexpected attention processor name: {name}")
        processors[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=rank,
        )
    unet.set_attn_processor(processors)
    return AttnProcsLayers(unet.attn_processors)


def set_attention_dropout(unet: UNet2DConditionModel, probability: float) -> None:
    """Match D-TRAK: set every Dropout module inside an attention block to p."""

    for name, module in unet.named_modules():
        if "attn" in name and isinstance(module, torch.nn.Dropout):
            module.p = probability


def checkpoint_epochs(num_epochs: int, num_checkpoints: int) -> set[int]:
    """Return exactly ``num_checkpoints`` approximately even, 1-based epochs."""

    if num_checkpoints < 1:
        raise ValueError("num_checkpoints must be at least 1")
    if num_checkpoints > num_epochs:
        raise ValueError("num_checkpoints cannot exceed num_epochs")
    return {
        math.ceil(checkpoint_number * num_epochs / num_checkpoints)
        for checkpoint_number in range(1, num_checkpoints + 1)
    }


def resolve_resume_checkpoint(output_dir: Path, value: str) -> Path:
    """Resolve an explicit checkpoint path or the latest epoch checkpoint."""

    if value != "latest":
        path = Path(value)
        if path.is_absolute() or path.exists():
            return path
        return output_dir / path

    checkpoints = sorted(output_dir.glob("checkpoint-epoch-*"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-epoch-* directories found in {output_dir}")
    return checkpoints[-1]


def parse_args(argv=None) -> argparse.Namespace:
    raw_root = data_root() / "raw" / "artbench2"
    index_root = data_root() / "artbench2" / "indices"
    parser = argparse.ArgumentParser(description="Fine-tune miniSD on ArtBench-2 with LoRA.")
    parser.add_argument("--model", default=ARTBENCH_MODEL_ID)
    parser.add_argument(
        "--model-revision",
        default=ARTBENCH_MODEL_REVISION,
        help="Hugging Face model commit, tag, or branch (paper default is pinned)",
    )
    parser.add_argument("--metadata-csv", type=Path, default=raw_root / "ArtBench-10.csv")
    parser.add_argument("--image-root", type=Path, default=raw_root / "artbench-10-imagefolder")
    parser.add_argument(
        "--index-path", type=Path, default=index_root / "idx-train.pkl"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_root() / "artbench2" / "model_training",
    )
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--attention-dropout", type=float, default=0.1)
    parser.add_argument("--conditioning-dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="fp16")
    parser.add_argument("--num-checkpoints", type=int, default=25)
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Full Accelerator state directory, or 'latest'.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    missing = [
        path
        for path in (args.metadata_csv, args.image_root, args.index_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"ArtBench-2 training input not found: {missing[0]}. "
            "Run experiments/artbench2/00_prepare_dataset.py first."
        )
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    pretrained = {"revision": args.model_revision}
    tokenizer = CLIPTokenizer.from_pretrained(
        args.model, subfolder="tokenizer", **pretrained
    )
    scheduler = DDPMScheduler.from_pretrained(
        args.model, subfolder="scheduler", **pretrained
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.model, subfolder="text_encoder", **pretrained
    )
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae", **pretrained)
    unet = UNet2DConditionModel.from_pretrained(
        args.model, subfolder="unet", **pretrained
    )
    empty_input_ids = tokenizer(
        "",
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids[0]

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # Match the official ordering: frozen base modules use inference precision,
    # while the subsequently created trainable LoRA parameters stay float32.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    lora_layers = create_lora_layers(unet, args.rank)
    set_attention_dropout(unet, args.attention_dropout)

    dataset = IndexedArtBenchDataset(
        args.metadata_csv,
        args.image_root,
        args.index_path,
        tokenizer,
        args.resolution,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    optimizer = torch.optim.AdamW(
        lora_layers.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(
        len(dataloader)
        / (accelerator.num_processes * args.gradient_accumulation_steps)
    )
    total_steps = updates_per_epoch * args.epochs
    epochs_to_checkpoint = checkpoint_epochs(args.epochs, args.num_checkpoints)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=math.ceil(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    lora_layers, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        lora_layers, optimizer, dataloader, lr_scheduler
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    first_epoch = 0
    global_step = 0
    if args.resume_from_checkpoint:
        resume_path = resolve_resume_checkpoint(
            args.output_dir, args.resume_from_checkpoint
        )
        state_path = resume_path / "training_state.json"
        if not state_path.is_file():
            raise FileNotFoundError(
                f"Missing {state_path}; cannot determine the exact resume position"
            )
        accelerator.load_state(str(resume_path))
        training_state = json.loads(state_path.read_text(encoding="utf-8"))
        first_epoch = int(training_state["completed_epoch"])
        global_step = int(training_state["global_step"])
        accelerator.print(
            f"Resumed full training state from {resume_path} "
            f"at completed_epoch={first_epoch} global_step={global_step}"
        )

    if accelerator.is_main_process:
        (args.output_dir / "pretrained_model.json").write_text(
            json.dumps(
                {
                    "model": args.model,
                    "revision": args.model_revision,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        trainable_parameters = {
            name: list(parameter.shape)
            for name, parameter in unet.named_parameters()
            if parameter.requires_grad
        }
        (args.output_dir / "trainable_parameters.json").write_text(
            json.dumps(trainable_parameters, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"trainable_parameters={sum(p.numel() for p in unet.parameters() if p.requires_grad):,} "
            f"effective_batch_size={args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes} "
            f"warmup_steps={math.ceil(total_steps * 0.1)} "
            f"checkpoint_epochs={sorted(epochs_to_checkpoint)}"
        )
    progress_bar = tqdm(
        total=total_steps,
        initial=global_step,
        desc="Training",
        dynamic_ncols=True,
        disable=not accelerator.is_local_main_process,
    )
    for epoch in range(first_epoch, args.epochs):
        unet.train()
        epoch_loss = 0.0
        update_loss = 0.0
        micro_batches_in_update = 0
        for batch in dataloader:
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(
                    accelerator.device, dtype=weight_dtype
                )
                input_ids = batch["input_ids"]
                if args.conditioning_dropout > 0:
                    drop_condition = (
                        torch.rand(input_ids.shape[0], device=input_ids.device)
                        < args.conditioning_dropout
                    )
                    input_ids = input_ids.clone()
                    input_ids[drop_condition] = empty_input_ids.to(input_ids.device)
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device,
                ).long()
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                hidden_states = text_encoder(input_ids)[0]
                prediction = unet(noisy_latents, timesteps, hidden_states).sample
                if scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif scheduler.config.prediction_type == "v_prediction":
                    target = scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unsupported prediction type: {scheduler.config.prediction_type}"
                    )
                loss = F.mse_loss(prediction.float(), target.float(), reduction="mean")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_layers.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            epoch_loss += loss.detach().float().item()
            update_loss += loss.detach().float().item()
            micro_batches_in_update += 1
            if accelerator.sync_gradients:
                global_step += 1
                mean_update_loss = update_loss / micro_batches_in_update
                progress_bar.update(1)
                progress_bar.set_postfix(
                    epoch=f"{epoch + 1}/{args.epochs}",
                    loss=f"{mean_update_loss:.4f}",
                    lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
                )
                update_loss = 0.0
                micro_batches_in_update = 0

        if accelerator.is_main_process:
            mean_loss = epoch_loss / max(len(dataloader), 1)
            print(f"epoch={epoch + 1}/{args.epochs} step={global_step} loss={mean_loss:.6f}")
        completed_epoch = epoch + 1
        if completed_epoch in epochs_to_checkpoint:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                checkpoint_path = (
                    args.output_dir / f"checkpoint-epoch-{completed_epoch:04d}"
                )
                accelerator.save_state(str(checkpoint_path))
                (checkpoint_path / "training_state.json").write_text(
                    json.dumps(
                        {
                            "completed_epoch": completed_epoch,
                            "global_step": global_step,
                            "pretrained_model": args.model,
                            "pretrained_model_revision": args.model_revision,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )

    progress_bar.close()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(unet).save_attn_procs(args.output_dir / "final")
        print(f"Saved final LoRA weights to {args.output_dir / 'final'}")


if __name__ == "__main__":
    main()


