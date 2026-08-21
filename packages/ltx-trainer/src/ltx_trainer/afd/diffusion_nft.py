"""DiffusionNFT on-policy student update (Sec. 3.2, Eq. 3–5)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.afd.config import AFDConfig


def linear_noise_schedule(t: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Rectified-flow style: α_t = 1−t, σ_t = t."""
    alpha = 1.0 - t
    sigma = t
    alpha_dot = -torch.ones_like(t)
    sigma_dot = torch.ones_like(t)
    return alpha, sigma, alpha_dot, sigma_dot


def _broadcast_time(t: Tensor, x: Tensor) -> Tensor:
    while t.dim() < x.dim():
        t = t.unsqueeze(-1)
    return t


def forward_noisy_state(x0: Tensor, t: Tensor, noise: Tensor | None = None) -> Tensor:
    """Eq. (3): x_t = α_t x̂_0 + σ_t ε."""
    if noise is None:
        noise = torch.randn_like(x0)
    alpha, sigma, _, _ = linear_noise_schedule(t)
    alpha = _broadcast_time(alpha, x0)
    sigma = _broadcast_time(sigma, x0)
    return alpha * x0 + sigma * noise


def forward_velocity_target(x0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
    """Eq. (3): v = α̇ x̂_0 + σ̇ ε."""
    _, _, alpha_dot, sigma_dot = linear_noise_schedule(t)
    alpha_dot = _broadcast_time(alpha_dot, x0)
    sigma_dot = _broadcast_time(sigma_dot, x0)
    return alpha_dot * x0 + sigma_dot * noise


def nft_velocity_operators(v_theta: Tensor, beta: float) -> tuple[Tensor, Tensor]:
    """v^+ and v^− with stop-gradient (Sec. 3.2)."""
    v_sg = v_theta.detach()
    v_plus = (1.0 - beta) * v_sg + beta * v_theta
    v_minus = (1.0 + beta) * v_sg - beta * v_theta
    return v_plus, v_minus


def diffusion_nft_loss(
    v_theta: Tensor,
    v_target: Tensor,
    weights: Tensor,
    *,
    beta: float = 0.1,
) -> Tensor:
    """Eq. (4): weighted positive/negative flow-matching."""
    v_plus, v_minus = nft_velocity_operators(v_theta, beta)
    w = weights.view(-1, *([1] * (v_theta.dim() - 1)))
    pos = w * (v_plus - v_target).pow(2)
    neg = (1.0 - w) * (v_minus - v_target).pow(2)
    return (pos + neg).mean()


def prior_regularization(
    v_theta: Tensor,
    v_ref: Tensor,
    weights: Tensor,
) -> Tensor:
    """L_prior: weighted MSE to frozen reference velocity (Eq. 5)."""
    w = weights.view(-1, *([1] * (v_theta.dim() - 1)))
    return (w * (v_theta - v_ref).pow(2)).mean()


def afd_student_loss(
    v_theta: Tensor,
    v_target: Tensor,
    v_ref: Tensor,
    weights: Tensor,
    cfg: AFDConfig,
) -> Tensor:
    """Eq. (5): L_AFD = L_NFT + λ_prior L_prior."""
    lnft = diffusion_nft_loss(v_theta, v_target, weights, beta=cfg.nft_beta)
    lprior = prior_regularization(v_theta, v_ref, weights)
    return lnft + cfg.prior_weight * lprior


class VelocityFieldStub(nn.Module):
    """Minimal velocity field for smoke tests."""

    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        self.dim = dim
        self.net = nn.Linear(dim + 1, dim)

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        b = x_t.shape[0]
        flat = x_t.reshape(b, -1)
        if flat.shape[1] < self.dim:
            flat = torch.nn.functional.pad(flat, (0, self.dim - flat.shape[1]))
        flat = flat[:, : self.dim]
        inp = torch.cat([flat, t.reshape(b, 1)], dim=-1)
        v = self.net(inp)
        return v.reshape(b, *x_t.shape[1:])
