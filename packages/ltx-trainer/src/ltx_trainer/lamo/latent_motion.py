"""Latent motion encoding and macro drift (Sec. 3.1, Eq. 1–4)."""

from __future__ import annotations

import torch
from torch import Tensor


def latent_delta(z: Tensor, *, tau: int = 2, dim: int = -4) -> Tensor:
    """τ-step latent change Δτ z := z(i+τ) − z(i) (Eq. 1).

    Expects time as dimension ``dim`` (default: batch layout B,T,C,H,W → dim=-4).
  """
    if tau < 1:
        raise ValueError("tau must be >= 1")
    return z.narrow(dim, tau, z.size(dim) - tau) - z.narrow(dim, 0, z.size(dim) - tau)


def macro_drift(delta: Tensor, *, spatial_dims: tuple[int, ...] = (-2, -1)) -> Tensor:
    """Channel-wise macro drift μ = E_{H,W}[Δτ z] (Eq. 4).

    Returns shape matching ``delta`` with spatial dims collapsed to length 1.
    """
    return delta.mean(dim=spatial_dims, keepdim=True)


def macro_drift_vector(delta: Tensor, *, spatial_dims: tuple[int, ...] = (-2, -1)) -> Tensor:
    """μ as a C-vector (no broadcast dims): mean over H,W per channel."""
    while delta.dim() > 1 and (delta.dim() - len(spatial_dims)) > 1:
        if delta.shape[-1] == 1 and -1 in spatial_dims:
            delta = delta.squeeze(-1)
        elif delta.shape[-2] == 1 and -2 in spatial_dims:
            delta = delta.squeeze(-2)
        else:
            break
    return delta.mean(dim=spatial_dims)


def empirical_macro_drift(z: Tensor, *, tau: int = 2, dim: int = -4) -> Tensor:
    """Parameter-free batch estimator: channel mean of Δτ z (Sec. 3.2)."""
    delta = latent_delta(z, tau=tau, dim=dim)
    return macro_drift_vector(delta, spatial_dims=(-2, -1) if delta.dim() >= 2 else ())


def select_strongest_motion_index(z: Tensor, *, tau: int = 2, dim: int = -4) -> int:
    """t* = argmax_t ||b_t||_2 for heatmaps (Appendix A, Eq. 11)."""
    delta = latent_delta(z, tau=tau, dim=dim)
    # delta indexed along time dim 0 when z is T,C,H,W
    if dim != 0 and z.dim() == 4:
        delta = latent_delta(z, tau=tau, dim=0)
    drifts = []
    for t in range(delta.size(0)):
        bt = macro_drift_vector(delta[t : t + 1]).squeeze(0)
        drifts.append(bt.norm().item())
    return int(max(range(len(drifts)), key=lambda i: drifts[i]))
