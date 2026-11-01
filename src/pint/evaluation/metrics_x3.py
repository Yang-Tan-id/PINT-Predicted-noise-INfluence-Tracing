from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pint.query_gradients.common import timestep_grid


def _final_checkpoint(root: Path) -> Path:
    paths = sorted((root / "base" / "prompted").glob("epoch_*.pt"))
    if not paths:
        raise FileNotFoundError(f"no x3 checkpoint under {root}")
    return paths[-1]


def evaluate(model_dir: Path, reference_dir: Path, query_manifest: Path, output: Path) -> None:
    import torch
    import torch.nn.functional as functional
    from tqdm.auto import tqdm

    from pint.training.backends import x3_model

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    target_payload = torch.load(_final_checkpoint(model_dir), map_location="cpu")
    reference_payload = torch.load(_final_checkpoint(reference_dir), map_location="cpu")
    vocab = target_payload["vocab"]

    def build(payload):
        model = x3_model.CondEpsModel(
            3, len(vocab), int(payload["base_ch"]), int(payload["time_dim"])
        ).to(device).eval()
        model.load_state_dict(payload["ema_model_state"], strict=True)
        return model

    target, reference = build(target_payload), build(reference_payload)
    records = json.loads(query_manifest.read_text(encoding="utf-8"))
    schedule = x3_model.make_linear_schedule(
        int(target_payload.get("T", 1000)), device=device
    )
    simple_timesteps = tuple(
        int(value) for value in timestep_grid("x3", schedule.T).tolist()
    )
    simple_loss, deviation, query_ids = [], [], []
    with torch.no_grad():
        for record in tqdm(records, desc="Evaluating LDS true f", unit="query", leave=False):
            query_id = int(record["query_id"])
            query_dir = Path(record["dir"])
            condition = torch.zeros((1, len(vocab)), device=device)
            for label in record["labels"]:
                condition[0, vocab[label]] = 1.0
            trajectory = torch.from_numpy(np.load(query_dir / "trajectory_xt.npy")).to(device=device, dtype=torch.float32)
            timesteps = torch.from_numpy(np.load(query_dir / "trajectory_t.npy")).to(device=device, dtype=torch.long)
            conditions = condition.expand(len(timesteps), -1)
            delta = target(trajectory[:, 0], timesteps, conditions) - reference(
                trajectory[:, 0], timesteps, conditions
            )
            deviation.append(float(delta.square().flatten(1).sum(1).mean().item()))
            endpoint = torch.from_numpy(np.load(query_dir / "final_state.npy")).to(
                device=device, dtype=torch.float32
            )
            losses = []
            for timestep in simple_timesteps:
                t = torch.tensor([timestep], device=device, dtype=torch.long)
                for mc_index in range(4):
                    generator = torch.Generator(device=str(device))
                    generator.manual_seed(
                        90_000_000 + query_id * 10_000 + timestep * 10 + mc_index
                    )
                    noise = torch.randn(
                        endpoint.shape, generator=generator, device=device,
                        dtype=endpoint.dtype,
                    )
                    noisy = x3_model.q_sample(endpoint, t, noise, schedule)
                    losses.append(functional.mse_loss(target(noisy, t, condition), noise))
            simple_loss.append(float(torch.stack(losses).mean().item()))
            query_ids.append(query_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        query_ids=np.asarray(query_ids, dtype=np.int32),
        simple_loss=np.asarray(simple_loss, dtype=np.float64),
        trajectory_deviation=np.asarray(deviation, dtype=np.float64),
        parameter_source=np.asarray("ema"),
        simple_loss_timesteps=np.asarray(simple_timesteps, dtype=np.int32),
        simple_loss_mc=np.asarray(4, dtype=np.int32),
        trajectory_reduction=np.asarray("mean_snapshot_l2_squared"),
    )
