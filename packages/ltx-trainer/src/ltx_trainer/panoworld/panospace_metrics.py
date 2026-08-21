"""PanoSpace-Bench metrics (Eq. 11–16)."""

from __future__ import annotations

import torch
from torch import Tensor


def multiple_choice_accuracy(pred: Tensor, target: Tensor) -> float:
    """Eq. 11 — exact choice match."""
    return float((pred == target).float().mean().item())


def _yaw_interval(yaw: Tensor, xfov: Tensor) -> tuple[Tensor, Tensor]:
    half = xfov / 2.0
    return yaw - half, yaw + half


def _interval_len(lo: Tensor, hi: Tensor) -> Tensor:
    return (hi - lo).clamp(min=0.0)


def _yaw_overlap_len(
    lo1: Tensor,
    hi1: Tensor,
    lo2: Tensor,
    hi2: Tensor,
) -> Tensor:
    """Yaw overlap with ±360° wrap (degrees)."""
    overlaps: list[Tensor] = []
    for shift in (-360.0, 0.0, 360.0):
        s_lo2 = lo2 + shift
        s_hi2 = hi2 + shift
        inter_lo = torch.maximum(lo1, s_lo2)
        inter_hi = torch.minimum(hi1, s_hi2)
        overlaps.append(_interval_len(inter_lo, inter_hi))
    return torch.stack(overlaps, dim=0).max(dim=0).values


def bf_ov_iou(pred: Tensor, target: Tensor) -> Tensor:
    """
    Angular IoU between BFOV boxes [..., 4] = yaw, pitch, xfov, yfov (degrees).

    Eq. 14–15; invalid predictions (non-positive extent) → 0 IoU.
    """
    py, pp, px, ph = pred.unbind(-1)
    ty, tp, tx, th = target.unbind(-1)
    valid = (px > 0) & (ph > 0) & (tx > 0) & (th > 0)
    p_lo, p_hi = _yaw_interval(py, px)
    t_lo, t_hi = _yaw_interval(ty, tx)
    p_plo, p_phi = pp - ph / 2, pp + ph / 2
    t_plo, t_phi = tp - th / 2, tp + th / 2
    yaw_inter = _yaw_overlap_len(p_lo, p_hi, t_lo, t_hi)
    pitch_inter = _interval_len(torch.maximum(p_plo, t_plo), torch.minimum(p_phi, t_phi))
    inter = yaw_inter * pitch_inter
    p_area = _interval_len(p_lo, p_hi) * _interval_len(p_plo, p_phi)
    t_area = _interval_len(t_lo, t_hi) * _interval_len(t_plo, t_phi)
    union = p_area + t_area - inter
    iou = inter / union.clamp(min=1e-6)
    return torch.where(valid, iou, torch.zeros_like(iou))


def bf_ov_miou(pred: Tensor, target: Tensor) -> float:
    """Eq. 16 — mean angular IoU."""
    if pred.numel() == 0:
        return 0.0
    return float(bf_ov_iou(pred, target).mean().item())
