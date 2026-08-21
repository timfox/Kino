"""Latent flow matching for joint audio-video (Eq. 1–2)."""

from __future__ import annotations

import torch
from torch import Tensor


def corrupt_latents(z_v: Tensor, z_a: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Eq. (1): x_t = (1-t)z + t·ε."""
    eps_v = torch.randn_like(z_v)
    eps_a = torch.randn_like(z_a)
    t_b = t.view(-1, *([1] * (z_v.dim() - 1)))
    x_v = (1 - t_b) * z_v + t_b * eps_v
    x_a = (1 - t_b) * z_a + t_b * eps_a
    return x_v, x_a, eps_v, eps_a


def flow_matching_loss(
    pred_v: Tensor,
    pred_a: Tensor,
    z_v: Tensor,
    z_a: Tensor,
    eps_v: Tensor,
    eps_a: Tensor,
) -> Tensor:
    """Eq. (2): L_DiT velocity target ε − z for both modalities."""
    target_v = eps_v - z_v
    target_a = eps_a - z_a
    return torch.nn.functional.mse_loss(pred_v, target_v) + torch.nn.functional.mse_loss(pred_a, target_a)
