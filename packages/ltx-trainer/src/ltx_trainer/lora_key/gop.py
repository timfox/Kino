"""Gradient Orthogonal Projection — Eqs. 13–14, § IV-B."""

from __future__ import annotations

import torch
from torch import Tensor


def projection_coefficient(g_wm: Tensor, g_sem: Tensor, *, eps: float = 1e-8) -> float:
    """α = ⟨g_wm, g_sem⟩ / (‖g_sem‖² + ε)."""
    dot = float(torch.dot(g_wm.flatten(), g_sem.flatten()).item())
    denom = float(torch.dot(g_sem.flatten(), g_sem.flatten()).item()) + eps
    return dot / denom


def gradient_orthogonal_projection(g_wm: Tensor, g_sem: Tensor, *, eps: float = 1e-8) -> Tensor:
    """g_proj = g_wm − α g_sem — remove semantic-conflicting watermark component."""
    alpha = projection_coefficient(g_wm, g_sem, eps=eps)
    return g_wm - alpha * g_sem


def cosine_similarity_flat(a: Tensor, b: Tensor, *, eps: float = 1e-8) -> float:
    a_f = a.flatten()
    b_f = b.flatten()
    return float(torch.dot(a_f, b_f).item() / (a_f.norm() * b_f.norm() + eps))
