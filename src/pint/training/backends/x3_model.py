import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import List, Sequence, Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader



# You said you have this:
from .x3_dataset import ColorGridDataset

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # CPU only; if you later use GPU, also set cuda seeds.
    torch.use_deterministic_algorithms(False)  # keep training sane; DDIM sampling will still be deterministic with eta=0.
    
    
@dataclass
class DiffusionSchedule:
    T: int
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor
    sqrt_alpha_bars: torch.Tensor
    sqrt_one_minus_alpha_bars: torch.Tensor

def make_linear_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02, device="cpu") -> DiffusionSchedule:
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    sqrt_alpha_bars = torch.sqrt(alpha_bars)
    sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)
    return DiffusionSchedule(
        T=T,
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        sqrt_alpha_bars=sqrt_alpha_bars,
        sqrt_one_minus_alpha_bars=sqrt_one_minus_alpha_bars,
    )
    
    
def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    t: (B,) integer timesteps
    returns: (B, dim)
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, device=t.device, dtype=torch.float32) / (half - 1)
    )
    # (B, half)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb



class CondEpsModel(nn.Module):
    def __init__(self, in_ch: int, cond_dim: int, base_ch: int = 64, time_dim: int = 128):
        super().__init__()
        self.time_dim = time_dim

        # Embed time and cond into a shared vector
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, base_ch),
            nn.SiLU(),
            nn.Linear(base_ch, base_ch),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, base_ch),
            nn.SiLU(),
            nn.Linear(base_ch, base_ch),
        )

        # Convolutional trunk
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
        )
        self.block3 = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
        )
        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, in_ch, 3, padding=1)

        # Project embedding to channel-wise bias
        self.emb_to_bias = nn.Linear(base_ch, base_ch)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x: (B,C,H,W)
        t: (B,) int64
        cond: (B,cond_dim)
        returns eps_pred: (B,C,H,W)
        """
        # (B, time_dim)
        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        # (B, base_ch)
        emb = self.time_mlp(t_emb) + self.cond_mlp(cond)
        bias = self.emb_to_bias(emb).unsqueeze(-1).unsqueeze(-1)  # (B, base_ch, 1, 1)

        h = self.in_conv(x)
        # Add conditioning bias at multiple depths (cheap & effective)
        h = h + bias
        h = h + self.block1(h)
        h = h + bias
        h = h + self.block2(h)
        h = h + bias
        h = h + self.block3(h)

        h = self.out_norm(h)
        h = F.silu(h)
        return self.out_conv(h)


def q_sample(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, sched: DiffusionSchedule) -> torch.Tensor:
    """
    x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*noise
    """
    # gather per-batch scalars
    a = sched.sqrt_alpha_bars[t].view(-1, 1, 1, 1)
    b = sched.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)
    return a * x0 + b * noise


@torch.no_grad()
def ddim_sample(
    model: nn.Module,
    sched: DiffusionSchedule,
    cond: torch.Tensor,
    shape: Tuple[int, int, int, int],  # (B,C,H,W)
    seed: int,
    steps: int = 50,
    eta: float = 0.0,
    device: str = "cpu",
    x_T: Optional[torch.Tensor] = None,
    save_steps: Optional[Sequence[int]] = None,  # NEW
) -> List[torch.Tensor]:
    """
    Deterministic DDIM when eta=0.

    NEW: returns snapshots at requested step indices.

    Step index definition (0..steps-1):
      - 0        = initial state (before any DDIM update), i.e., starting x
      - k (1.. ) = state after k updates
      - steps-1  = final state after all updates

    If save_steps is None -> default [0, steps-1] (initial + final).
    """

    assert eta == 0.0, "Only eta=0 (deterministic DDIM) is supported currently."

    set_seed(seed)
    B, C, H, W = shape

    # init x
    if x_T is None:
        x = torch.randn(shape, device=device)
    else:
        x = x_T.to(device)
        if tuple(x.shape) != tuple(shape):
            raise ValueError(f"x_T shape {tuple(x.shape)} != expected {tuple(shape)}")

    # timesteps used by DDIM
    T = sched.T
    ts = torch.linspace(T - 1, 0, steps, device=device).long()

    # which step indices to save
    if save_steps is None:
        save_steps = [0, steps - 1]

    save_steps = [int(k) for k in save_steps]
    for k in save_steps:
        if k < 0 or k > steps - 1:
            raise ValueError(f"save_steps contains {k}, but valid range is [0, {steps-1}]")

    save_set = set(save_steps)
    saved: Dict[int, torch.Tensor] = {}

    # step 0 snapshot (initial)
    if 0 in save_set:
        saved[0] = x.detach().clone()

    # DDIM updates; after i-th update, we are at step index i+1
    for i in range(len(ts) - 1):
        t = ts[i].repeat(B)
        t_prev = ts[i + 1].item()

        eps = model(x, t, cond)
        alpha_bar_t = sched.alpha_bars[t].view(-1, 1, 1, 1)
        alpha_bar_prev = sched.alpha_bars[t_prev].view(1, 1, 1, 1)

        x0_pred = (x - torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
        x = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1.0 - alpha_bar_prev) * eps

        step_idx = i + 1
        if step_idx in save_set:
            saved[step_idx] = x.detach().clone()

    # return snapshots in the same order the user requested
    return [saved[k] for k in save_steps]


def main():
    # -------------------------
    # Config (edit these)
    # -------------------------
    save_name    =  "49_100000"
    csv_path     = f"generated_database/{save_name}.csv"   # <-- set your file
    grid_size    = 3                               # 3 for 3x3, 9 for 9x9, etc
    batch_size   = 256
    epochs       = 350
    lr           = 2e-4
    T            = 2000                             # diffusion steps
    seed         = 0                               # controls training randomness + DDIM starting noise
    sample_every = 10
    ddim_steps   = 50

    device = "cpu"  # you said no GPU
    set_seed(seed)

    # Dataset: supports grid_size if you added it; otherwise it will ignore
    if "grid_size" in ColorGridDataset.__init__.__code__.co_varnames:
        ds = ColorGridDataset(csv_path, grid_size=grid_size)
    else:
        ds = ColorGridDataset(csv_path)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    # Peek shapes
    x0, cond0 = ds[0]
    C, H, W = x0.shape
    cond_dim = cond0.numel()
    print(f"Dataset: N={len(ds)}, image=(C,H,W)=({C},{H},{W}), cond_dim={cond_dim}, vocab_size={cond_dim}")

    sched = make_linear_schedule(T, device=device)
    model = CondEpsModel(in_ch=C, cond_dim=cond_dim, base_ch=64, time_dim=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []

        for x0, cond in loader:
            x0 = x0.to(device)  # (B,C,H,W), in [0,1]
            cond = cond.to(device)

            B = x0.size(0)
            t = torch.randint(0, sched.T, (B,), device=device, dtype=torch.long)
            noise = torch.randn_like(x0)
            xt = q_sample(x0, t, noise, sched)

            eps_pred = model(xt, t, cond)
            loss = F.mse_loss(eps_pred, noise)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            losses.append(loss.item())

        avg_loss = float(np.mean(losses))
        print(f"Epoch {epoch:03d}/{epochs}  loss={avg_loss:.6f}")

        # Quick deterministic DDIM sample check
        if (epoch % sample_every) == 0:
            model.eval()
            x_real, c_real = ds[random.randrange(len(ds))]
            c_real = c_real.unsqueeze(0).to(device)  # (1,cond_dim)

            x_gen = ddim_sample(
                model=model,
                sched=sched,
                cond=c_real,
                shape=(1, C, H, W),
                seed=seed,
                steps=ddim_steps,
                eta=0.0,
                device=device,
            )[1]

            print(
                f"  Sample stats: min={x_gen.min().item():.3f}, max={x_gen.max().item():.3f}, mean={x_gen.mean().item():.3f}"
            )

    ckpt = {
        "model_state": model.state_dict(),
        "T": T,
        "grid_size": grid_size,
        "cond_dim": cond_dim,
        "seed": seed,
        "vocab": getattr(ds, "vocab", None),
    }
    # usage:
    save_path = next_checkpoint_path("models", save_name)  # e.g. models/49_10_001.pt
    torch.save(ckpt, save_path)
    print(f"Saved checkpoint to {save_path}")


def next_checkpoint_path(models_dir: str, base_name: str, ext: str = ".pt") -> str:
    os.makedirs(models_dir, exist_ok=True)
    i = 1
    while True:
        path = os.path.join(models_dir, f"{base_name}_{i:03d}{ext}")
        if not os.path.exists(path):
            return path
        i += 1


if __name__ == "__main__":
    main()
