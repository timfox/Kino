"""Geometry metrics for representation divergence (Sec. 2, Eq. 2–3)."""

from __future__ import annotations

import torch
from torch import Tensor


def reconstruction_residual(flow_g: Tensor, basis_u: Tensor) -> float:
    r"""``R_res(G | U) = ‖G - Π_U G‖²_F / ‖G‖²_F`` (Eq. 2).

    ``flow_g`` and ``basis_u`` are ``[N, D]`` row features. ``basis_u`` defines the PCA subspace of flow U.
    """
    if flow_g.numel() == 0:
        return 0.0
    # Orthonormal basis via QR on centered U
    u = basis_u - basis_u.mean(dim=0, keepdim=True)
    q, _ = torch.linalg.qr(u.T, mode="reduced")
    proj = flow_g @ q @ q.T
    resid = flow_g - proj
    num = float(resid.pow(2).sum().item())
    den = float(flow_g.pow(2).sum().item()) + 1e-8
    return num / den


def effective_rank(x: Tensor, eps: float = 1e-8) -> float:
    """Stable rank proxy: exp(entropy of normalized singular values)."""
    if x.numel() == 0:
        return 0.0
    x_c = x - x.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(x_c)
    p = s / (s.sum() + eps)
    ent = -(p * (p + eps).log()).sum()
    return float(torch.exp(ent).item())


def effective_rank_increment(h_u: Tensor, h_g: Tensor, h_u_g: Tensor) -> float:
    r"""``ΔER(H_G ; X | H_U) = ER(H_{U,G}) - ER(H_U)`` (Eq. 3)."""
    return effective_rank(h_u_g) - effective_rank(h_u)
