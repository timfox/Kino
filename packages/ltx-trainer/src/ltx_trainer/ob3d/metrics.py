"""Evaluation metrics for CPE, NVS, and 3D reconstruction (Sec. 3.5, D.2–D.4)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _align_mask(mask: Tensor, ref: Tensor) -> Tensor:
    if mask.shape == ref.shape:
        return mask
    if mask.dim() == 2 and ref.dim() == 4:
        return mask.view(1, 1, *mask.shape).expand_as(ref)
    return mask.expand_as(ref)


def depth_rmse(pred: Tensor, gt: Tensor, mask: Tensor | None = None) -> Tensor:
    if mask is None:
        mask = torch.ones_like(gt, dtype=torch.bool)
    mask = _align_mask(mask, gt)
    err = (pred - gt).pow(2)
    return torch.sqrt(err[mask].mean().clamp_min(1e-8))


def depth_mae(pred: Tensor, gt: Tensor, mask: Tensor | None = None) -> Tensor:
    if mask is None:
        mask = torch.ones_like(gt, dtype=torch.bool)
    mask = _align_mask(mask, gt)
    return (pred - gt).abs()[mask].mean()


def depth_abs_rel(pred: Tensor, gt: Tensor, mask: Tensor | None = None, eps: float = 1e-6) -> Tensor:
    if mask is None:
        mask = torch.ones_like(gt, dtype=torch.bool)
    mask = _align_mask(mask, gt)
    return ((pred - gt).abs() / gt.clamp_min(eps))[mask].mean()


def depth_delta125(
    pred: Tensor,
    gt: Tensor,
    mask: Tensor | None = None,
    ratio: float = 1.25,
) -> Tensor:
    if mask is None:
        mask = torch.ones_like(gt, dtype=torch.bool)
    mask = _align_mask(mask, gt)
    thresh = torch.maximum(gt / pred, pred / gt)
    return (thresh[mask] < ratio).float().mean()


def sky_mask(depth: Tensor, threshold: float) -> Tensor:
    """Exclude sky / extremely far depth (Sec. 3.5)."""
    return depth < threshold


def psnr(pred: Tensor, gt: Tensor, eps: float = 1e-8) -> Tensor:
    mse = F.mse_loss(pred, gt)
    return 10.0 * torch.log10(1.0 / mse.clamp_min(eps))


def ssim_stub(pred: Tensor, gt: Tensor) -> Tensor:
    """Structural similarity stub (window-free mean correlation)."""
    p = pred.flatten()
    g = gt.flatten()
    c = ((p - p.mean()) * (g - g.mean())).mean()
    v1 = p.var()
    v2 = g.var()
    return (2 * c + 1e-3) / (v1 + v2 + 1e-3)


def relative_rotation_error(r_est: Tensor, r_gt: Tensor) -> float:
    """RRA in degrees (angular error between rotation matrices)."""
    r = r_est @ r_gt.transpose(-1, -2)
    trace = r.trace().clamp(-1, 3)
    ang = torch.acos(((trace - 1) / 2).clamp(-1, 1))
    return float(torch.rad2deg(ang).item())


def relative_translation_error(t_est: Tensor, t_gt: Tensor) -> float:
    """RTA in degrees between translation directions."""
    te = t_est / t_est.norm().clamp_min(1e-6)
    tg = t_gt / t_gt.norm().clamp_min(1e-6)
    cos = (te * tg).sum().clamp(-1, 1)
    return float(torch.rad2deg(torch.acos(cos)).item())


def auc_at_threshold(rra: list[float], rta: list[float], deg: float = 5.0) -> float:
    if not rra:
        return 0.0
    ok = sum(1 for a, b in zip(rra, rta, strict=True) if a < deg and b < deg)
    return ok / len(rra)


def ate_rmse(est_positions: Tensor, gt_positions: Tensor) -> float:
    """ATE after positions are aligned (stub: same scale)."""
    err = (est_positions - gt_positions).pow(2).sum(dim=-1).sqrt()
    return float(torch.sqrt(err.pow(2).mean()).item())
