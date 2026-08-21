"""Depth calibration protocol (Sec. 3): learnable scale λ."""

from __future__ import annotations

import torch
from torch import Tensor


def learn_scaling_lambda(pred_depth: Tensor, gt_depth: Tensor) -> float:
    """
    λ = Σ_i d_i d_gt,i / Σ_i d_i²  (training landmarks, gravity-aligned).
    """
    p = pred_depth.detach().float().flatten()
    g = gt_depth.detach().float().flatten()
    if p.numel() == 0:
        return 1.0
    num = (p * g).sum()
    den = (p * p).sum().clamp_min(1e-8)
    return float(num / den)


def landmark_mse(
    pred_depth: Tensor,
    gt_depth: Tensor,
    lambda_scale: float,
) -> float:
    """Mean squared error on test landmarks after scaling."""
    scaled = lambda_scale * pred_depth.float()
    return float(((scaled - gt_depth.float()) ** 2).mean().item())


def split_landmarks(
    n: int,
    train_ratio: float,
    *,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Return train and test index tensors."""
    perm = torch.randperm(n, generator=generator)
    n_train = max(1, int(n * train_ratio))
    return perm[:n_train], perm[n_train:]
