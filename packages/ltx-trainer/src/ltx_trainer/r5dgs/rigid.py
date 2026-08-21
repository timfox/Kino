"""Rigid-body constrained extrapolation (Sec. II-B, Eq. 3–5)."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.r5dgs.quaternion import quat_mul, quat_rotate_vector


def representative_indices(x: Tensor, group_ids: Tensor, num_groups: int) -> Tensor:
    """One representative index per group: Gaussian closest to centroid (Sec. II-B).

    ``x`` (N, 3), ``group_ids`` (N,) int in [0, num_groups).
    Returns ``rep_idx`` (N,) where ``rep_idx[i]`` is the rep index for Gaussian ``i``'s group.
    """
    n = x.shape[0]
    rep = torch.zeros(num_groups, dtype=torch.long, device=x.device)
    for g in range(num_groups):
        m = group_ids == g
        if not m.any():
            rep[g] = 0
            continue
        pts = x[m]
        c = pts.mean(dim=0)
        dist = (pts - c).norm(dim=-1)
        local = dist.argmin()
        idx_global = torch.nonzero(m, as_tuple=False).squeeze(-1)[local]
        rep[g] = idx_global
    # map each Gaussian to its group's rep index (global index)
    return rep[group_ids]


def canonical_offsets(x: Tensor, rep_idx: Tensor) -> Tensor:
    """Eq. (3): ``o_i = x_i - x_rep[g(i)]``."""
    return x - x[rep_idx]


def propagate_rigid_positions(
    x_def: Tensor,
    x_vel_at_rep: Tensor,
    q_vel_at_rep: Tensor,
    o: Tensor,
    rep_idx: Tensor,
) -> Tensor:
    """Eq. (4): extrapolated centers from deformation state + rep-only TRD integration.

    ``x_vel_at_rep[i]`` / ``q_vel_at_rep[i]`` are the integrated pose for the **representative**
    of Gaussian ``i``'s group (broadcast to all members). ``o`` are canonical offsets (Eq. 3).
    """
    xdr = x_def[rep_idx]
    dx = (x_vel_at_rep - xdr) + quat_rotate_vector(q_vel_at_rep, o) - o
    return x_def + dx


def propagate_rigid_quaternions(
    q_vel_at_rep: Tensor,
    delta_q_def: Tensor,
) -> Tensor:
    """Eq. (5): ``q_out_i = q_vel_rk ⊗ Δq_def_i`` (Hamilton product)."""
    return quat_mul(q_vel_at_rep, delta_q_def)


def knn_indices_bruteforce(x: Tensor, k: int) -> Tensor:
    """``x`` (N, 3) → neighbor indices (N, k) excluding self (use next nearest if duplicate)."""
    n = x.shape[0]
    d = torch.cdist(x, x)
    d.fill_diagonal_(float("inf"))
    _, idx = d.topk(k, dim=-1, largest=False)
    return idx
