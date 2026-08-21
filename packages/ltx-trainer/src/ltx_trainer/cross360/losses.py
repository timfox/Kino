"""Multi-scale depth supervision (Eq. 4–6, BerHu for S2D3D)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def sobel_grad(depth: Tensor) -> Tensor:
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=depth.dtype, device=depth.device).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(depth, kx, padding=1)
    gy = F.conv2d(depth, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def mse_depth_loss(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    diff = (pred - target) ** 2
    if mask is not None:
        diff = diff * mask
        return diff.sum() / mask.sum().clamp_min(1.0)
    return diff.mean()


def grad_depth_loss(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    gp, gt = sobel_grad(pred), sobel_grad(target)
    diff = (gp - gt).abs()
    if mask is not None:
        diff = diff * mask
        return diff.sum() / mask.sum().clamp_min(1.0)
    return diff.mean()


def total_depth_loss(
    preds: list[Tensor],
    targets: list[Tensor],
    *,
    mask: Tensor | None = None,
) -> dict[str, Tensor]:
    l_mse = sum(mse_depth_loss(p, t, mask) for p, t in zip(preds, targets)) / len(preds)
    l_grad = sum(grad_depth_loss(p, t, mask) for p, t in zip(preds, targets)) / len(preds)
    return {"L_mse": l_mse, "L_grad": l_grad, "L_total": l_mse + l_grad}
