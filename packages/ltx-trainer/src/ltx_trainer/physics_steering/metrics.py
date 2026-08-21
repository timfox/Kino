"""Evaluation metrics (Sec. 3.6)."""

from __future__ import annotations

from torch import Tensor


def flip_rate(base_pred: Tensor, steered_pred: Tensor) -> float:
    """Fraction of samples whose probe classification changes."""
    if base_pred.shape != steered_pred.shape:
        raise ValueError("base_pred and steered_pred must match shape")
    return float((base_pred != steered_pred).float().mean().item())


def score_delta(base_prob: Tensor, steered_prob: Tensor) -> float:
    """Mean ΔP(impossible)."""
    return float((steered_prob - base_prob).mean().item())


def directional_purity(delta_f: Tensor, cav: Tensor) -> float:
    """DP = cos(Δf, v); batched mean if (B, D)."""
    v = cav / (cav.norm() + 1e-8)
    if delta_f.ndim == 1:
        d = delta_f / (delta_f.norm() + 1e-8)
        return float((d @ v).item())
    d = delta_f / (delta_f.norm(dim=-1, keepdim=True) + 1e-8)
    return float((d * v).sum(dim=-1).mean().item())


def representation_drift(delta_f: Tensor) -> float:
    """RD = ||Δf||_2 (mean over batch if batched)."""
    if delta_f.ndim == 1:
        return float(delta_f.norm().item())
    return float(delta_f.norm(dim=-1).mean().item())
