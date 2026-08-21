"""KLD loop-closure confidence bias (Sec. IV-B, Eq. 3–5)."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.ground_texture_slam.histogram import baseline_histogram, channel_histogram


def kld_channel(p: Tensor, q: Tensor, *, eps: float = 1e-12) -> float:
    """KL divergence D(p||q) for discrete distributions."""
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    return float((p * torch.log(p / q)).sum())


def kld_rgb(image: Tensor, baseline: Tensor, *, grayscale: bool = False) -> float:
    """Average KLD across channels (Eq. 4)."""
    p = channel_histogram(image if not grayscale else image.mean(dim=0, keepdim=True))
    q = baseline if not grayscale else baseline[:1]
    if p.shape[0] != q.shape[0]:
        q = q[: p.shape[0]]
    scores = [kld_channel(p[c], q[c]) for c in range(p.shape[0])]
    return sum(scores) / len(scores)


def scale_covariance_kld(cov: Tensor, kld: float) -> Tensor:
    """Σ* = Σ × (KLD + 1) (Eq. 5)."""
    return cov * (kld + 1.0)


def build_baseline_from_session(images: list[Tensor], *, grayscale: bool = False) -> Tensor:
    base = baseline_histogram(images)
    if grayscale:
        return base.mean(dim=0, keepdim=True)
    return base
