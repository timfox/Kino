"""Flow matching + shortcut ANC losses (Eq. 7.3, 7.7–7.8)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.pixelwizard.config import PixelWizardConfig
from ltx_trainer.pixelwizard.shortcut import consistency_target, shortcut_consistency_loss


def flow_matching_loss(velocity: Tensor, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
    """Rectified-flow target u_t = x1 - x0 (Eq. 7.2–7.3 proxy)."""
    t = t.view(-1, *([1] * (x0.dim() - 1)))
    x_t = (1 - t) * x0 + t * x1
    target = x1 - x0
    return F.mse_loss(velocity, target)


def shortcut_training_loss(
    model_fn,
    x_t: Tensor,
    t: Tensor,
    delta_t: int,
    cfg: PixelWizardConfig | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Self-consistency L_sc with adaptive noise-span weight."""
    cfg = cfg or PixelWizardConfig()
    dt = torch.full((x_t.shape[0],), float(delta_t), device=x_t.device, dtype=x_t.dtype)
    v_dt = model_fn(x_t, t, dt)
    x_next = x_t + (dt.view(-1, 1, 1, 1) / cfg.num_diffusion_steps) * v_dt
    v_next = model_fn(x_next, t + dt / cfg.num_diffusion_steps, dt)
    pred_2 = model_fn(x_t, t, dt * 2)
    tgt = consistency_target(v_dt, v_next, x_t, t, dt).detach()
    t_idx = int(t[0].item() * (cfg.num_diffusion_steps - 1)) if t.numel() else 600
    loss = shortcut_consistency_loss(pred_2, tgt, t_idx, delta_t, cfg.noise_span_power)
    return loss, {"lambda_anc": float(loss.detach() / (F.mse_loss(pred_2, tgt).detach() + 1e-8))}
