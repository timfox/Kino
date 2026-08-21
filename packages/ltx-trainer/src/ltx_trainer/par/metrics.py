"""Evaluation metrics stubs (FID, FAED, CLIP, DS)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def discontinuity_score(pano: Tensor) -> Tensor:
    """
    DS: mean L1 seam at left/right ERP boundary (lower is better).
    pano [B,C,H,W]
    """
    left = pano[..., :, :1]
    right = pano[..., :, -1:]
    return (left - right).abs().mean()


def fid_stub(real: Tensor, fake: Tensor) -> Tensor:
    """Feature-distance stub (lower is better)."""
    fr = real.flatten(1).mean(dim=0)
    ff = fake.flatten(1).mean(dim=0)
    return (fr - ff).pow(2).mean().sqrt()


def clip_score_stub(_text: str, image: Tensor) -> Tensor:
    """Placeholder alignment score in [0, 1] scale."""
    return torch.tensor(0.5 + 0.1 * image.mean())


def psnr(pred: Tensor, gt: Tensor, eps: float = 1e-8) -> Tensor:
    mse = F.mse_loss(pred, gt)
    return 10.0 * torch.log10(1.0 / mse.clamp_min(eps))
