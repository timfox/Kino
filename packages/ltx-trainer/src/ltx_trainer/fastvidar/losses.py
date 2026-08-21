"""ERP-weighted depth loss (Sec. III-D, Eq. 16–18)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.fastvidar.erp_projection import erp_latitude_weight


def huber_residual(diff: Tensor, delta: float = 1.0) -> Tensor:
    abs_d = diff.abs()
    quad = torch.minimum(abs_d, torch.tensor(delta, device=diff.device, dtype=diff.dtype))
    lin = abs_d - quad
    return 0.5 * quad**2 + delta * lin


def erp_data_loss(
    pred: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    confidence: Tensor | None = None,
    height: int | None = None,
    huber_delta: float = 1.0,
) -> Tensor:
    """Eq. (16) with latitude weights w(v) = cos φ."""
    h = height or pred.shape[-2]
    w_row = erp_latitude_weight(h, device=pred.device).view(1, h, 1)
    conf = confidence if confidence is not None else torch.ones_like(pred)
    residual = huber_residual(pred - target, huber_delta)
    return (w_row * mask * conf * residual).sum() / mask.sum().clamp(min=1.0)


def erp_gradient_loss(
    pred: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    num_scales: int = 1,
    huber_delta: float = 1.0,
) -> Tensor:
    """Eq. (17) multi-scale ERP gradients."""
    total = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    p = pred.unsqueeze(0) if pred.dim() == 2 else pred
    t = target.unsqueeze(0) if target.dim() == 2 else target
    m = mask.unsqueeze(0) if mask.dim() == 2 else mask
    for scale_i in range(num_scales):
        h = p.shape[-2]
        w_row = erp_latitude_weight(h, device=p.device).view(1, h, 1)
        dp_du = p[..., :, 1:] - p[..., :, :-1]
        dt_du = t[..., :, 1:] - t[..., :, :-1]
        mu = m[..., :, 1:] * m[..., :, :-1]
        total = total + (w_row * mu * huber_residual(dp_du - dt_du, huber_delta)).mean()
        dp_dv = p[..., 1:, :] - p[..., :-1, :]
        dt_dv = t[..., 1:, :] - t[..., :-1, :]
        mv = m[..., 1:, :] * m[..., :-1, :]
        w_row_v = erp_latitude_weight(h, device=p.device).view(1, h, 1)[:, 1:, :]
        total = total + (w_row_v * mv * huber_residual(dp_dv - dt_dv, huber_delta)).mean()
        if scale_i + 1 < num_scales and min(p.shape[-2], p.shape[-1]) > 2:
            p = F.avg_pool2d(p, 2)
            t = F.avg_pool2d(t, 2)
            m = F.avg_pool2d(m.float(), 2)
    return total / max(num_scales, 1)


def depth_objective(
    pred: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    lambda_grad: float = 1.0,
    huber_delta: float = 1.0,
) -> dict[str, Tensor]:
    """Eq. (18) per-frame objective."""
    ld = erp_data_loss(pred, target, mask, huber_delta=huber_delta)
    lg = erp_gradient_loss(pred, target, mask, huber_delta=huber_delta)
    return {"loss": ld + lambda_grad * lg, "data": ld, "grad": lg}
