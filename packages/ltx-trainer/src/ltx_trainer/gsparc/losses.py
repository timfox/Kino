"""Training objectives (Eq. 4–6)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def channel_mse_loss(
    h_hat: Tensor,
    h_gt: Tensor,
) -> Tensor:
    """Eq. (4): L_ch on real and imaginary parts."""
    return F.mse_loss(h_hat[0], h_gt[0]) + F.mse_loss(h_hat[1], h_gt[1])


def ssim_stub(x: Tensor, y: Tensor) -> Tensor:
    """Structural similarity proxy (1 − mean abs diff of local means)."""
    mx = x.mean(dim=(-2, -1), keepdim=True)
    my = y.mean(dim=(-2, -1), keepdim=True)
    vx = ((x - mx) ** 2).mean(dim=(-2, -1))
    vy = ((y - my) ** 2).mean(dim=(-2, -1))
    c1, c2 = 0.01**2, 0.03**2
    num = (2 * mx * my + c1) * (2 * (x * y).mean(dim=(-2, -1)) + c2)
    den = (mx**2 + my**2 + c1) * (vx + vy + c2)
    return (num / den.clamp(min=1e-6)).mean()


def spectrum_loss(
    z_hat: Tensor,
    z_gt: Tensor,
    *,
    lambda_l1: float = 0.2,
) -> Tensor:
    """Eq. (5): L_sp = (1−λ) L1 + λ (1−SSIM)."""
    mag_hat = torch.sqrt(z_hat[0] ** 2 + z_hat[1] ** 2 + 1e-8)
    mag_gt = torch.sqrt(z_gt[0] ** 2 + z_gt[1] ** 2 + 1e-8)
    l1 = F.l1_loss(mag_hat, mag_gt)
    ssim_val = ssim_stub(mag_hat.unsqueeze(0), mag_gt.unsqueeze(0))
    return (1.0 - lambda_l1) * l1 + lambda_l1 * (1.0 - ssim_val)


def confidence_raw(g_out: Tensor) -> Tensor:
    """C = 1 + exp(g_φ(x_rx)), C > 1."""
    return 1.0 + torch.exp(g_out.squeeze(-1))


def confidence_weighted_loss(
    task_loss: Tensor,
    confidence: Tensor,
    *,
    alpha: float = 0.3,
) -> Tensor:
    """Eq. (6): L_conf = C · L_task − α log C."""
    return confidence * task_loss - alpha * torch.log(confidence.clamp(min=1.0 + 1e-6))


def normalize_confidence(c: Tensor) -> Tensor:
    """C̃ = (C − C_min) / (C_max − C_min) ∈ [0, 1]."""
    cmin = c.min()
    cmax = c.max()
    if (cmax - cmin).abs() < 1e-8:
        return torch.zeros_like(c)
    return (c - cmin) / (cmax - cmin)
