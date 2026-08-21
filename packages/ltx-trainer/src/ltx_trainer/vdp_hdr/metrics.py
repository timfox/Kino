"""Scale-invariant Q-style metrics (Cao et al. / Talegaonkar eval)."""

from __future__ import annotations

import torch
from torch import Tensor


def _align_scale(pred: Tensor, gt: Tensor, *, eps: float = 1e-8) -> Tensor:
    pf = pred.reshape(-1).float()
    gf = gt.reshape(-1).float()
    denom = (pf * pf).sum().clamp(min=eps)
    s = (pf * gf).sum() / denom
    return pred * s


def _normalize_peak(x: Tensor, eps: float = 1e-8) -> Tensor:
    peak = x.max().clamp(min=eps)
    return x / peak


def q_mae(pred: Tensor, gt: Tensor) -> float:
    p = _normalize_peak(_align_scale(pred, gt))
    g = _normalize_peak(gt)
    return float((p - g).abs().mean())


def q_psnr(pred: Tensor, gt: Tensor) -> float:
    p = _normalize_peak(_align_scale(pred, gt))
    g = _normalize_peak(gt)
    mse = float(((p - g) ** 2).mean())
    if mse < 1e-12:
        return 99.0
    return float(10.0 * torch.log10(torch.tensor(1.0 / mse)))
