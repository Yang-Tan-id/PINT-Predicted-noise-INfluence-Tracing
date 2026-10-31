"""Reconstruct batched AdamW directions without mutating checkpoint state."""

from __future__ import annotations

import math
from typing import Sequence


def adamw_full_direction_batch(
    parameters: Sequence,
    gradients: Sequence,
    optimizer,
    *,
    max_grad_norm: float | None,
):
    """Return each example's next full AdamW update with LR factored out."""
    import torch

    if len(parameters) != len(gradients) or not gradients:
        raise ValueError("parameters and non-empty gradients must have the same length")
    batch_size = gradients[0].shape[0]
    if any(gradient.shape[0] != batch_size for gradient in gradients):
        raise ValueError("all gradient leaves must share a leading example axis")
    if max_grad_norm is not None and max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive or None")
    if max_grad_norm is None:
        clip_scales = torch.ones(batch_size, device=gradients[0].device)
    else:
        norms = torch.sqrt(
            sum(
                gradient.detach().float().reshape(batch_size, -1).square().sum(dim=1)
                for gradient in gradients
            )
        )
        clip_scales = (max_grad_norm / norms.clamp_min(1e-12)).clamp_max(1.0)

    groups = {
        id(parameter): group
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    directions = []
    learning_rates = set()
    for parameter, gradient in zip(parameters, gradients):
        group = groups.get(id(parameter))
        if group is None:
            raise KeyError("parameter is absent from optimizer param_groups")
        if group.get("amsgrad", False):
            raise NotImplementedError("AMSGrad is not supported")
        beta1, beta2 = group["betas"]
        eps = float(group["eps"])
        weight_decay = float(group["weight_decay"])
        learning_rates.add(float(group["lr"]))
        state = optimizer.state.get(parameter, {})
        step_value = state.get("step", 0)
        step = int(step_value.item() if torch.is_tensor(step_value) else step_value) + 1
        exp_avg = state.get("exp_avg", torch.zeros_like(parameter)).detach()
        exp_avg_sq = state.get("exp_avg_sq", torch.zeros_like(parameter)).detach()
        view_shape = (batch_size,) + (1,) * parameter.ndim
        grad = gradient.detach().to(dtype=parameter.dtype) * clip_scales.to(
            dtype=parameter.dtype
        ).view(view_shape)
        next_avg = exp_avg.unsqueeze(0).mul(beta1).add(grad, alpha=1.0 - beta1)
        next_avg_sq = exp_avg_sq.unsqueeze(0).mul(beta2).addcmul(
            grad, grad, value=1.0 - beta2
        )
        correction1 = 1.0 - beta1**step
        correction2 = 1.0 - beta2**step
        denominator = next_avg_sq.sqrt().div(math.sqrt(correction2)).add(eps)
        adaptive = next_avg.div(correction1).div(denominator)
        directions.append(
            adaptive.add(parameter.detach().unsqueeze(0), alpha=weight_decay).neg()
        )
    if len(learning_rates) != 1:
        raise ValueError("attribution requires one shared LR across AdamW groups")
    return directions, learning_rates.pop()
