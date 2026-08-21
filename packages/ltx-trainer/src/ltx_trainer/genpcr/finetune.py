"""Few-shot Match-ControlNet finetuning loss (Sec. 3.3.3, 3.4.4, Eq. 5, 8, 11)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def denoising_matching_loss(
    epsilon_pred: Tensor,
    epsilon: Tensor,
) -> Tensor:
    """L = E[||ε − ε_θ(x_t, t, c, d)||²] (Eq. 5, 8, 11)."""
    return F.mse_loss(epsilon_pred, epsilon)


def sample_coupled_noise(
    shape: tuple[int, ...],
    *,
    device: torch.device | None = None,
) -> Tensor:
    return torch.randn(shape, device=device)


def forward_diffusion_stub(
    x0: Tensor,
    t: int,
    *,
    num_steps: int = 1000,
) -> tuple[Tensor, Tensor]:
    """
    Reparameterize x_t = sqrt(ᾱ_t) x0 + sqrt(1-ᾱ_t) ε for a linear schedule stub.
    Returns (x_t, ε).
    """
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    t = max(0, min(t, num_steps - 1))
    alpha_bar = 1.0 - (t + 1) / num_steps
    alpha_bar = max(alpha_bar, 1e-4)
    eps = torch.randn_like(x0)
    x_t = alpha_bar**0.5 * x0 + (1.0 - alpha_bar) ** 0.5 * eps
    return x_t, eps
