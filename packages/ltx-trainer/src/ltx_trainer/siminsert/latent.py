"""Flow-matching latents and Latent Refresh (Sec. III-C, Eq. 6–7)."""

from __future__ import annotations

import torch
from torch import Tensor


def flow_matching_forward(
    x0: Tensor,
    t: float | Tensor,
    epsilon: Tensor | None = None,
) -> Tensor:
    """x_t = (1 - t) x0 + t ε (Eq. 2, 6)."""
    if epsilon is None:
        epsilon = torch.randn_like(x0)
    if not isinstance(t, Tensor):
        t = torch.tensor(t, device=x0.device, dtype=x0.dtype)
    t = t.reshape(-1, *([1] * (x0.dim() - 1)))
    return (1.0 - t) * x0 + t * epsilon


def reconstruction_latent(
    x0: Tensor,
    t: float | Tensor,
    epsilon: Tensor,
) -> Tensor:
    """Reconstruction-path latent at timestep t (Eq. 2)."""
    return flow_matching_forward(x0, t, epsilon)


def latent_refresh(
    x_edited: Tensor,
    x0: Tensor,
    mask: Tensor,
    t: float | Tensor,
    epsilon: Tensor,
) -> Tensor:
    """Blend edited latent with forward-noised background (Eq. 7).

    mask: 1 = edited region, 0 = background (preserve).
    """
    x_bg = flow_matching_forward(x0, t, epsilon)
    m = _expand_mask(mask, x_edited)
    return m * x_edited + (1.0 - m) * x_bg


def _expand_mask(mask: Tensor, target: Tensor) -> Tensor:
    m = mask.to(dtype=target.dtype, device=target.device)
    while m.dim() < target.dim():
        m = m.unsqueeze(-1)
    if m.shape[-1] == 1 and target.shape[-1] > 1:
        return m
    if m.shape[1] == target.shape[1]:
        return m
    if m.dim() == 3 and target.dim() == 4:
        return m.unsqueeze(-1)
    return m
