"""Training losses (Eq. 2, 6, 7) and render proxy (3DGS-style L1 + SSIM weight)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def render_loss_l1_ssim(pred: Tensor, target: Tensor, *, lambda_dssim: float = 0.2) -> Tensor:
    """``L_render = (1 - λ) L1 + λ (1 - SSIM)`` — SSIM as global scalar proxy over (B,3,H,W)."""
    l1 = (pred - target).abs().mean()
    c1, c2 = 0.01**2, 0.03**2
    mu_x = pred.mean(dim=(1, 2, 3), keepdim=True)
    mu_y = target.mean(dim=(1, 2, 3), keepdim=True)
    var_x = pred.var(dim=(1, 2, 3), unbiased=False, keepdim=True)
    var_y = target.var(dim=(1, 2, 3), unbiased=False, keepdim=True)
    cov = ((pred - mu_x) * (target - mu_y)).mean(dim=(1, 2, 3), keepdim=True)
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2) + 1e-8)
    ssim = ssim.mean()
    return (1.0 - lambda_dssim) * l1 + lambda_dssim * (1.0 - ssim)


def loss_obj_2d(logits: Tensor, target_class: Tensor) -> Tensor:
    """Cross-entropy for composited identity map (Sec. II-A)."""
    return F.cross_entropy(logits, target_class)


def loss_3d_neighbor_kl(
    identity: Tensor,
    neighbor_idx: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """KL regularization among k-NN in feature space (Sec. II-A).

    ``identity`` (N, D), ``neighbor_idx`` (N, k) int64 in [0, N).
    """
    p = F.softmax(identity, dim=-1).clamp_min(eps)
    nbr = identity[neighbor_idx]
    p_mean = nbr.mean(dim=1)
    q = F.softmax(p_mean, dim=-1).clamp_min(eps)
    return (p * (p.log() - q.log())).sum(dim=-1).mean()


def loss_rigid_distance(
    x_before: Tensor,
    x_after: Tensor,
    rep_idx: Tensor,
) -> Tensor:
    """Eq. (6): penalize change in distance to each Gaussian's representative.

    ``rep_idx`` (N,) int64: ``rep_idx[i]`` is index of representative for Gaussian ``i``.
    """
    if rep_idx.shape != x_before.shape[:1]:
        raise ValueError("rep_idx must be (N,) same as number of Gaussians")
    xb_r = x_before[rep_idx]
    xa_r = x_after[rep_idx]
    db = (x_before - xb_r).norm(dim=-1)
    da = (x_after - xa_r).norm(dim=-1)
    return (da - db).pow(2).mean()


def loss_majority_consistency(
    logits: Tensor,
    neighbor_idx: Tensor,
) -> Tensor:
    """Eq. (7): MSE between class distribution and mean of neighbors."""
    p = F.softmax(logits, dim=-1)
    pn = F.softmax(logits[neighbor_idx], dim=-1).mean(dim=1)
    return (p - pn).pow(2).sum(dim=-1).mean()


def total_loss_r5dgs(
    l_render: Tensor,
    l_obj: Tensor,
    l_3d: Tensor,
    *,
    lambda_obj: float,
    lambda_3d: float,
    iteration: int,
    tau_reg: int,
    l_rigid: Tensor | None = None,
    l_major: Tensor | None = None,
    lambda_rigid: float = 0.5,
    lambda_maj: float = 0.5,
    t_rigid: int = 10_000,
) -> Tensor:
    """Eq. (2) with optional L_rigid / L_major gated by iteration and ``tau_reg``."""
    l = l_render + lambda_obj * l_obj
    if iteration % tau_reg == 0:
        l = l + lambda_3d * l_3d
        if l_major is not None:
            l = l + lambda_maj * l_major
    if l_rigid is not None and iteration >= t_rigid:
        l = l + lambda_rigid * l_rigid
    return l
