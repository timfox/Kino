"""Differentiable rendering losses + curvature-aware remeshing (Sec. 3.2, Eq. 1–6)."""

from __future__ import annotations

import torch
from torch import Tensor


def silhouette_loss(pred: Tensor, target: Tensor) -> Tensor:
    """L_sil (Eq. 1 / 35)."""
    return torch.nn.functional.mse_loss(pred, target)


def normal_map_loss(pred: Tensor, target: Tensor, *, mask: Tensor | None = None) -> Tensor:
    """1 − n̂·n on foreground (Eq. 2 / 36)."""
    pred_n = torch.nn.functional.normalize(pred, dim=-1)
    tgt_n = torch.nn.functional.normalize(target, dim=-1)
    dot = (pred_n * tgt_n).sum(dim=-1)
    loss_map = 1.0 - dot
    if mask is not None:
        m = mask > 0.5
        if m.any():
            return loss_map[m].mean()
        return loss_map.mean()
    return loss_map.mean()


def mesh_objective(
    pred_sil: Tensor,
    tgt_sil: Tensor,
    pred_norm: Tensor,
    tgt_norm: Tensor,
    *,
    lambda_n: float = 1.0,
    lambda_s: float = 1.0,
) -> dict[str, Tensor]:
    """Φ(M) silhouette + normal terms (Eq. 3 / 34)."""
    l_sil = silhouette_loss(pred_sil, tgt_sil)
    l_n = normal_map_loss(pred_norm, tgt_norm, mask=tgt_sil)
    total = lambda_n * l_n + lambda_s * l_sil
    return {"total": total, "silhouette": l_sil, "normal": l_n}


def normal_rotation_rate(normal_map: Tensor) -> Tensor:
    """s(u,v) = ||∂n/∂u, ∂n/∂v||_2 proxy on H×W×3 grid."""
    if normal_map.ndim != 3:
        raise ValueError("normal_map must be (H, W, 3)")
    n = torch.nn.functional.normalize(normal_map, dim=-1)
    du = n[1:, :, :] - n[:-1, :, :]
    dv = n[:, 1:, :] - n[:, :-1, :]
    s_u = du.norm(dim=-1)
    s_v = dv.norm(dim=-1)
    h, w = n.shape[0], n.shape[1]
    rate = torch.zeros(h, w, device=n.device, dtype=n.dtype)
    rate[:-1, :] = torch.maximum(rate[:-1, :], s_u)
    rate[:, :-1] = torch.maximum(rate[:, :-1], s_v)
    return rate


def projected_target_edge_length(
    rotation_rate: Tensor,
    *,
    theta0: float = 0.15,
    pmin: float = 1.0,
    pmax: float = 32.0,
) -> Tensor:
    """p_tgt = clip(θ0 / s, pmin, pmax) (Eq. 4 / 40)."""
    s = rotation_rate.clamp(min=1e-4)
    return (theta0 / s).clamp(pmin, pmax)


def update_slack(
    slack: Tensor,
    velocity: Tensor,
    *,
    gain: float = 0.5,
    nu_ref: float = 0.02,
) -> Tensor:
    """ζ_k(i) ← max(0, ζ_{k-1} + g ν − ν_ref) (Eq. 5 / 44)."""
    return torch.clamp(slack + gain * velocity - nu_ref, min=0.0)


def reference_edge_length(
    l_curv: Tensor,
    slack: Tensor,
    *,
    lmin: float,
    lmax: float,
) -> Tensor:
    """L_ref = clip(L_curv + ζ, Lmin, Lmax) (Eq. 6 / 45)."""
    return (l_curv + slack).clamp(lmin, lmax)
