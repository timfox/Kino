"""Joint intensity histogram symmetry (Sec. IV-D, Eq. 6–10)."""

from __future__ import annotations

import torch
from torch import Tensor


def joint_intensity_histogram(img_a: Tensor, img_b: Tensor, *, bins: int = 256) -> Tensor:
    """2D joint histogram H (Eq. 6) on aligned grayscale images."""
    if img_a.dim() == 3:
        img_a = img_a.mean(dim=0)
    if img_b.dim() == 3:
        img_b = img_b.mean(dim=0)
    ia = (img_a.clamp(0, 1) * (bins - 1)).long().flatten()
    ib = (img_b.clamp(0, 1) * (bins - 1)).long().flatten()
    h = torch.zeros(bins, bins)
    for a, b in zip(ia.tolist(), ib.tolist()):
        h[a, b] += 1.0
    return h / h.sum().clamp(min=1.0)


def jih_symmetry_score(h: Tensor) -> float:
    """JIH score in [0,1] (Eq. 7–9)."""
    hs = 0.5 * (h + h.T)
    ha = 0.5 * (h - h.T)
    num = hs.norm() - ha.norm()
    den = hs.norm() + ha.norm() + 1.0
    return float(0.5 * (num / den))


def scale_covariance_jih(cov: Tensor, jih: float, *, eps: float = 1e-6) -> Tensor:
    """Σ* = Σ × 1/(JIH + ε) (Eq. 10)."""
    return cov / (jih + eps)
