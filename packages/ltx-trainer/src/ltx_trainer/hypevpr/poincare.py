"""Poincaré ball operations (Sec. 3.1–3.2, Eq. 3–5)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _sqrt_c(c: float) -> float:
    return math.sqrt(max(c, 1e-12))


def mobius_add(x: Tensor, y: Tensor, *, c: float = 1.0, eps: float = 1e-8) -> Tensor:
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = (1 + 2 * c * xy + c * c * x2 * y2).clamp_min(eps)
    return project(num / den, c=c)


def mobius_neg(x: Tensor) -> Tensor:
    return -x


def poincare_distance(x: Tensor, y: Tensor, *, c: float = 1.0, eps: float = 1e-8) -> Tensor:
    """Hyperbolic distance d_c (Eq. 3)."""
    diff = mobius_add(x, mobius_neg(y), c=c, eps=eps)
    norm = diff.norm(dim=-1).clamp_min(eps)
    sc = _sqrt_c(c)
    return (2.0 / sc) * torch.arctanh((sc * norm).clamp(max=1.0 - 1e-5))


def expmap0(v: Tensor, *, c: float = 1.0, eps: float = 1e-8) -> Tensor:
    """expc_0(v) (Eq. 4 at origin)."""
    vnorm = v.norm(dim=-1, keepdim=True).clamp_min(eps)
    sc = _sqrt_c(c)
    return torch.tanh(sc * vnorm) * v / (sc * vnorm)


def logmap0(x: Tensor, *, c: float = 1.0, eps: float = 1e-8) -> Tensor:
    """logc_0(x) (Eq. 5 at origin)."""
    xnorm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    sc = _sqrt_c(c)
    return (2.0 / sc) * torch.arctanh((sc * xnorm).clamp(max=1.0 - 1e-5)) * x / xnorm


def project(x: Tensor, *, c: float = 1.0, eps: float = 1e-5) -> Tensor:
    maxnorm = (1.0 - eps) / _sqrt_c(c)
    norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    return torch.where(norm > maxnorm, x * (maxnorm / norm), x)


def einstein_midpoint(vectors: Tensor, *, c: float = 1.0, eps: float = 1e-8) -> Tensor:
    """
    A_hyp over [N, D] (Eq. 16–17).
    """
    if vectors.shape[0] == 1:
        return vectors[0]
    sq = (vectors * vectors).sum(dim=-1)
    gamma = 1.0 / torch.sqrt((1.0 - c * sq).clamp_min(eps))
    weighted = (gamma.unsqueeze(-1) * vectors).sum(dim=0)
    return project(weighted / gamma.sum().clamp_min(eps), c=c)
