from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from pint.evaluation.common import EvaluationJob
from pint.training.model_versions import (
    ARTBENCH_CLIP_MODEL_ID,
    ARTBENCH_CLIP_MODEL_REVISION,
    ARTBENCH_DINO_MODEL,
    ARTBENCH_DINO_REPOSITORY,
    ARTBENCH_DINO_REVISION,
)


def _batches(values: Sequence, size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def add_perceptual_metrics(jobs: list[EvaluationJob], batch_size: int = 16) -> None:
    import lpips
    import torch
    from PIL import Image
    from torchvision import transforms
    from transformers import CLIPModel, CLIPProcessor
    from tqdm.auto import tqdm

    if not torch.cuda.is_available():
        raise RuntimeError("ArtBench-2 perceptual evaluation requires an NVIDIA GPU")
    device = torch.device("cuda:0")
    paths: list[Path] = []
    for job in jobs:
        assets = job.output_dir / "evaluation_assets"
        paths.extend((assets / "reference.png", assets / "candidate.png"))
    images = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            images.append(image.convert("RGB"))

    processor = CLIPProcessor.from_pretrained(
        ARTBENCH_CLIP_MODEL_ID, revision=ARTBENCH_CLIP_MODEL_REVISION
    )
    clip_model = CLIPModel.from_pretrained(
        ARTBENCH_CLIP_MODEL_ID, revision=ARTBENCH_CLIP_MODEL_REVISION
    ).half().eval().to(device)
    clip_parts = []
    for batch in tqdm(
        list(_batches(images, batch_size)), desc="ArtBench CLIP", unit="batch"
    ):
        pixels = processor(images=list(batch), return_tensors="pt")["pixel_values"]
        with torch.inference_mode():
            value = clip_model.get_image_features(
                pixel_values=pixels.to(device=device, dtype=torch.float16)
            ).float()
            clip_parts.append(torch.nn.functional.normalize(value, dim=-1).cpu().numpy())
    clip_features = np.concatenate(clip_parts)
    del clip_model
    torch.cuda.empty_cache()

    dino_model = torch.hub.load(
        f"{ARTBENCH_DINO_REPOSITORY}:{ARTBENCH_DINO_REVISION}",
        ARTBENCH_DINO_MODEL,
    ).eval().to(device)
    preprocess = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    dino_parts = []
    for batch in tqdm(
        list(_batches(images, batch_size)), desc="ArtBench DINO", unit="batch"
    ):
        values = torch.stack([preprocess(image) for image in batch]).to(device)
        with torch.inference_mode():
            value = torch.nn.functional.normalize(dino_model(values).float(), dim=-1)
            dino_parts.append(value.cpu().numpy())
    dino_features = np.concatenate(dino_parts)
    del dino_model
    torch.cuda.empty_cache()

    lpips_model = lpips.LPIPS(net="alex").eval().to(device)
    to_tensor = transforms.ToTensor()
    lpips_values = []
    with torch.inference_mode():
        for index in tqdm(range(len(jobs)), desc="ArtBench LPIPS", unit="model"):
            reference = (to_tensor(images[2 * index]) * 2.0 - 1.0).unsqueeze(0).to(device)
            candidate = (to_tensor(images[2 * index + 1]) * 2.0 - 1.0).unsqueeze(0).to(device)
            lpips_values.append(float(lpips_model(reference, candidate).item()))

    for index, job in enumerate(jobs):
        metrics_path = job.output_dir / "true_f.npz"
        with np.load(metrics_path, allow_pickle=False) as payload:
            values = {name: np.asarray(payload[name]) for name in payload.files}
        left, right = 2 * index, 2 * index + 1
        values["clip_distance"] = np.asarray(
            [1.0 - float(np.dot(clip_features[left], clip_features[right]))]
        )
        values["dino_distance"] = np.asarray(
            [1.0 - float(np.dot(dino_features[left], dino_features[right]))]
        )
        values["lpips"] = np.asarray([lpips_values[index]])
        np.savez_compressed(metrics_path, **values)
