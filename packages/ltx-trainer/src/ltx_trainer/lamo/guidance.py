"""Motion Prior Guidance at sampling (Eq. 8–9)."""

from __future__ import annotations

import torch
from torch import Tensor


def guidance_loss(
    x0_hat: Tensor,
    motion_field: Tensor,
    *,
    tau: int = 2,
    time_dim: int = 0,
) -> Tensor:
    """L_guide (Eq. 8): mean over frame pairs of ||x̂0(i) + f_φ − x̂0(i+τ)||^2."""
    n_pairs = x0_hat.size(time_dim) - tau
    if n_pairs <= 0:
        return torch.tensor(0.0, device=x0_hat.device, dtype=x0_hat.dtype)
    total = torch.tensor(0.0, device=x0_hat.device, dtype=x0_hat.dtype)
    for i in range(n_pairs):
        xi = x0_hat.select(time_dim, i)
        xj = x0_hat.select(time_dim, i + tau)
        field_i = motion_field.select(time_dim, i) if motion_field.dim() == x0_hat.dim() else motion_field
        pred_next = xi + field_i
        total = total + (pred_next - xj).pow(2).mean()
    return total / n_pairs


def guidance_active(step_index: int, num_steps: int, *, active_ratio: float = 0.8) -> bool:
    """1{s >= (1 − ρ) S} gate for Eq. 9 (s indexed from 0 at high noise)."""
    threshold = int((1.0 - active_ratio) * num_steps)
    return step_index >= threshold


def apply_motion_prior_guidance(
    eps_cfg: Tensor,
    x0_hat: Tensor,
    predictor: torch.nn.Module,
    cond: Tensor,
    *,
    step_index: int,
    num_steps: int,
    lambda_guide: float = 25.0,
    active_ratio: float = 0.8,
    tau: int = 2,
) -> Tensor:
    """ε̂_guided = ε̂_CFG − λ_guide · gate · ∇_{ε̂} L_guide (Eq. 9).

    ``x0_hat`` must retain grad w.r.t. ``eps_cfg`` for autograd guidance.
    """
    if not guidance_active(step_index, num_steps, active_ratio=active_ratio):
        return eps_cfg.detach() if not eps_cfg.requires_grad else eps_cfg

    x0_req = x0_hat
    if not x0_req.requires_grad:
        x0_req = x0_hat.detach().requires_grad_(True)

    fields = []
    t_len = x0_req.size(0)
    for i in range(max(0, t_len - tau)):
        zi = x0_req[i]
        fields.append(predictor(zi.unsqueeze(0), cond))
    if not fields:
        return eps_cfg
    motion_stack = torch.cat(fields, dim=0)
    loss = guidance_loss(x0_req, motion_stack, tau=tau, time_dim=0)
    grad = torch.autograd.grad(loss, eps_cfg, retain_graph=False, create_graph=False)[0]
    return eps_cfg - lambda_guide * grad
