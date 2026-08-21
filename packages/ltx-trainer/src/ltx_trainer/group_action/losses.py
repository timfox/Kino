"""Latent group-action losses (Eq. 10–13)."""

from __future__ import annotations

import torch
from torch import Tensor


def latent_mse(a: Tensor, b: Tensor) -> Tensor:
    """D_z(x, y) = ||x - y||_2^2 (Sec. 3.1)."""
    return (a - b).pow(2).mean()


def identity_loss(z_end: Tensor, z_start: Tensor) -> Tensor:
    """L_id (Eq. 10)."""
    return latent_mse(z_end, z_start)


def inverse_loss(z_end: Tensor, z_start: Tensor) -> Tensor:
    """L_inv: endpoint after forward–inverse should match z_start."""
    return latent_mse(z_end, z_start)


def composition_loss(z_a: Tensor, z_b: Tensor) -> Tensor:
    """L_comp: compatible decompositions share endpoint (Eq. 10)."""
    return latent_mse(z_a, z_b)


def group_action_loss(
    *,
    z_start: Tensor,
    z_id: Tensor | None = None,
    z_inv: Tensor | None = None,
    z_comp_a: Tensor | None = None,
    z_comp_b: Tensor | None = None,
    lambda_id: float = 1.0,
    lambda_inv: float = 1.0,
    lambda_comp: float = 1.0,
) -> Tensor:
    """Eq. (12): weighted sum of active constraint losses."""
    total = torch.tensor(0.0, device=z_start.device, dtype=z_start.dtype)
    if z_id is not None:
        total = total + lambda_id * identity_loss(z_id, z_start)
    if z_inv is not None:
        total = total + lambda_inv * inverse_loss(z_inv, z_start)
    if z_comp_a is not None and z_comp_b is not None:
        total = total + lambda_comp * composition_loss(z_comp_a, z_comp_b)
    return total
