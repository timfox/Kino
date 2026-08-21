"""Matting evaluation metrics (MAD, MSE, Grad, Conn)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def matting_mad(pred: Tensor, gt: Tensor) -> float:
    """Mean absolute difference × 1000 (common matting benchmark scale)."""
    return float((pred.detach() - gt.detach()).abs().mean() * 1000.0)


def matting_mse(pred: Tensor, gt: Tensor) -> float:
    return float((pred.detach() - gt.detach()).pow(2).mean() * 1000.0)


def gradient_error(pred: Tensor, gt: Tensor) -> float:
    """Spatial gradient L1 (Grad metric family)."""
    def g(x: Tensor) -> Tensor:
        gx = x[..., :, 1:] - x[..., :, :-1]
        gy = x[..., 1:, :] - x[..., :-1, :]
        return torch.cat([gx.flatten(1), gy.flatten(1)], dim=1)

    return float(F.l1_loss(g(pred), g(gt)).item() * 1000.0)


def connectivity_error(pred: Tensor, gt: Tensor, threshold: float = 0.1) -> float:
    """Connectivity error proxy: XOR of thresholded regions (Conn family, ×1000)."""
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    if gt.dim() == 3:
        gt = gt.unsqueeze(0)
    pm = (pred >= threshold).float()
    gm = (gt >= threshold).float()
    xor = (pm - gm).abs()
    # Penalize fragmented boundaries via morphological dilate (max-pool)
    k = 3
    pad = k // 2
    pm_d = F.max_pool2d(pm, k, stride=1, padding=pad)
    gm_d = F.max_pool2d(gm, k, stride=1, padding=pad)
    conn = 0.5 * xor.mean() + 0.5 * (pm_d - gm_d).abs().mean()
    return float(conn.item() * 1000.0)


def dtssd(
    pred_sequence: Tensor,
    gt_sequence: Tensor,
    *,
    trimap_unknown_width: int = 25,
) -> float:
    """
    Temporal stability (dtSSD, Erofeev et al. / VideoMatte family).

    Args:
        pred_sequence: ``[T, 1, H, W]`` or ``[T, H, W]`` predicted alphas.
        gt_sequence: matching ground truth.
    """
    if pred_sequence.dim() == 3:
        pred_sequence = pred_sequence.unsqueeze(1)
    if gt_sequence.dim() == 3:
        gt_sequence = gt_sequence.unsqueeze(1)
    t = pred_sequence.shape[0]
    if t < 2:
        return 0.0
    total = pred_sequence.new_tensor(0.0)
    for i in range(t - 1):
        tri = trimap_from_alpha(gt_sequence[i : i + 2].mean(dim=0, keepdim=False))
        if tri.dim() == 2:
            tri = tri.unsqueeze(0)
        unk = (tri > 0.05) & (tri < 0.95)
        if unk.dim() == 3 and unk.shape[0] == 1:
            unk = unk.squeeze(0)
        dp = (pred_sequence[i + 1] - pred_sequence[i]).abs()
        if unk.any():
            total = total + dp[unk].mean()
        else:
            total = total + dp.mean()
    return float(total.item() / (t - 1) * 1000.0)


def trimap_from_alpha(alpha: Tensor, unknown_width: int = 25) -> Tensor:
    """Re-export-friendly trimap for dtSSD (matches training loss bands)."""
    from ltx_trainer.cinematte.losses import trimap_from_alpha as _t

    return _t(alpha, unknown_width)
