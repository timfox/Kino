"""Simplex utilities for distribution-valued time series."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.cast.config import METRIC_EPS


def normalize_simplex(p: Tensor, *, eps: float = METRIC_EPS) -> Tensor:
    p = p.clamp(min=0.0)
    return p / p.sum(dim=-1, keepdim=True).clamp(min=eps)


def support_mean(p: Tensor) -> Tensor:
    """μ(p) = Σ_j j·p(j) for ordered supports."""
    idx = torch.arange(p.shape[-1], device=p.device, dtype=p.dtype)
    return (p * idx).sum(dim=-1)


def kl_divergence(p: Tensor, q: Tensor, *, eps: float = METRIC_EPS) -> Tensor:
    """KL(p‖q), reduced over last dim unless keepdim."""
    p = normalize_simplex(p, eps=eps)
    q = normalize_simplex(q, eps=eps)
    return (p * (torch.log(p + eps) - torch.log(q + eps))).sum(dim=-1)


def jensen_shannon(p: Tensor, q: Tensor, *, eps: float = METRIC_EPS) -> Tensor:
    """JSD(p, q)."""
    p = normalize_simplex(p, eps=eps)
    q = normalize_simplex(q, eps=eps)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, eps=eps) + 0.5 * kl_divergence(q, m, eps=eps)


def l1_distance(p: Tensor, q: Tensor) -> Tensor:
    return (normalize_simplex(p) - normalize_simplex(q)).abs().sum(dim=-1)


def w1_ordered(p: Tensor, q: Tensor) -> Tensor:
    """1-Wasserstein on ordered categorical support."""
    p = normalize_simplex(p)
    q = normalize_simplex(q)
    cdf_p = p.cumsum(dim=-1)
    cdf_q = q.cumsum(dim=-1)
    return (cdf_p - cdf_q).abs().sum(dim=-1)


def weighted_js(mixtures: list[Tensor], weights: Tensor) -> Tensor:
    """Weighted Jensen–Shannon divergence JS_π(u1,…,uK) for aliasing lower bound."""
    weights = weights / weights.sum()
    mixtures = [normalize_simplex(u) for u in mixtures]
    mixture = sum(w * u for w, u in zip(weights, mixtures))
    total = torch.zeros((), device=weights.device, dtype=weights.dtype)
    for w, u in zip(weights, mixtures):
        total = total + w * kl_divergence(u, mixture)
    return total
