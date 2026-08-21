"""Trajectory prediction metrics ADE / FDE (Sec. IV-B, Eq. 1–2)."""

from __future__ import annotations

import torch
from torch import Tensor


def ade(pred: Tensor, gt: Tensor) -> Tensor:
    """Average displacement error over prediction horizon."""
    if pred.shape != gt.shape:
        raise ValueError("pred and gt must have same shape (T, 2)")
    return torch.norm(pred - gt, dim=-1).mean()


def fde(pred: Tensor, gt: Tensor) -> Tensor:
    """Final displacement error at last timestep."""
    return torch.norm(pred[-1] - gt[-1])


def batch_ade_fde(pred: Tensor, gt: Tensor) -> tuple[float, float]:
    """Mean ADE/FDE over agents; shapes (N, T, 2)."""
    ades = [float(ade(pred[i], gt[i])) for i in range(pred.shape[0])]
    fdes = [float(fde(pred[i], gt[i])) for i in range(pred.shape[0])]
    return sum(ades) / len(ades), sum(fdes) / len(fdes)
