"""Flow-matching objective and ODE integration for WT-DiT."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def logit_normal_sample(batch: int, *, loc: float = 0.0, scale: float = 1.0, device=None) -> Tensor:
    u = torch.randn(batch, device=device) * scale + loc
    return torch.sigmoid(u)


def plateau_logit_normal_sample(batch: int, *, device=None) -> Tensor:
    """Plateaued schedule — more mass near t→0 for visible layer."""
    base = logit_normal_sample(batch, loc=-0.5, scale=0.8, device=device)
    return base * 0.85 + 0.05


def layer_aware_timestep(
    num_layers: int,
    batch: int,
    *,
    phase: str = "mixture",
    device=None,
) -> Tensor:
    """Sample diffusion times per layer (Sec. 3.3 / App. B)."""
    if phase == "independent":
        t0 = plateau_logit_normal_sample(batch, device=device)
        t_deep = logit_normal_sample(batch, device=device)
        rows = [t0]
        for _ in range(num_layers - 1):
            rows.append(t_deep)
        return torch.stack(rows, dim=1)
    u = torch.rand(batch, device=device)
    t_a = logit_normal_sample(batch, device=device)
    t_b = plateau_logit_normal_sample(batch, device=device)
    t = torch.where(u < 0.5, t_a, t_b)
    return t.unsqueeze(1).expand(batch, num_layers)


def interpolate_endpoint(x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
    while t.ndim < x0.ndim:
        t = t.unsqueeze(-1)
    return (1.0 - t) * x0 + t * x1


def flow_endpoint_loss(
    pred_x0: Tensor,
    target_x0: Tensor,
    alpha_mask: Tensor,
    *,
    lambda_mono: float = 0.1,
) -> Tensor:
    """Endpoint flow-matching loss L_FM + monotonicity (Eq. 3, 8)."""
    diff = (pred_x0 - target_x0) ** 2
    masked = diff * alpha_mask
    denom = alpha_mask.sum().clamp_min(1.0)
    l_fm = masked.sum() / denom
    z = pred_x0[..., 2]
    if z.shape[0] >= 2:
        violations = torch.relu(z[:-1] - z[1:])
        l_mono = (violations**2).mean()
    else:
        l_mono = pred_x0.new_zeros(())
    return l_fm + lambda_mono * l_mono


def velocity_from_endpoint(xt: Tensor, pred_x0: Tensor, t: Tensor) -> Tensor:
    t_safe = t.clamp_min(1e-3)
    while t_safe.ndim < xt.ndim:
        t_safe = t_safe.unsqueeze(-1)
    return (xt - pred_x0) / t_safe


def integrate_flow(
    model_fn,
    x1: Tensor,
    t_schedule: Tensor,
    **model_kw,
) -> Tensor:
    """Explicit Euler integration from t=1 toward t=0."""
    xt = x1
    steps = len(t_schedule) - 1
    for i in range(steps):
        t_cur = t_schedule[i]
        t_next = t_schedule[i + 1]
        dt = t_cur - t_next
        pred_x0 = model_fn(xt, t_cur, **model_kw)
        v = velocity_from_endpoint(xt, pred_x0, t_cur)
        xt = xt - dt * v
    return xt


def default_ode_schedule(num_steps: int, device=None) -> Tensor:
    return torch.linspace(1.0, 0.0, num_steps + 1, device=device)
