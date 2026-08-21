"""Cosine velocity schedule + masked discrete text diffusion (§3.3)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def cosine_schedule(t: Tensor) -> tuple[Tensor, Tensor]:
    """α_t = cos(πt/2), σ_t = sin(πt/2)."""
    half = 0.5 * math.pi * t
    return torch.cos(half), torch.sin(half)


def noisy_audio_latent(z0: Tensor, t: Tensor, eps: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
    if eps is None:
        eps = torch.randn_like(z0)
    alpha, sigma = cosine_schedule(t)
    while alpha.ndim < z0.ndim:
        alpha = alpha.unsqueeze(-1)
        sigma = sigma.unsqueeze(-1)
    zt = alpha * z0 + sigma * eps
    v_target = alpha * eps - sigma * z0
    return zt, v_target, eps


def mask_probability(tau: Tensor, eps: float = 1e-3) -> Tensor:
    """p_mask(τ) = (1 - ε)τ for masked text diffusion."""
    return (1.0 - eps) * tau


def apply_random_mask(
    token_ids: Tensor,
    tau: Tensor,
    *,
    mask_id: int,
    eps: float = 1e-3,
) -> tuple[Tensor, Tensor]:
    """Return corrupted tokens and binary mask (1 = predict)."""
    p = mask_probability(tau, eps=eps).view(-1, 1)
    keep = torch.rand_like(token_ids.float()) > p
    corrupted = torch.where(keep, token_ids, torch.full_like(token_ids, mask_id))
    mask = (~keep).long()
    return corrupted, mask


def text_loss_weight(tau: Tensor, eps: float = 1e-3) -> Tensor:
    """w(τ) = σ'(τ) / (exp(σ(τ)) - 1) with σ(τ) = -log(1 - (1-ε)τ)."""
    one_minus = 1.0 - (1.0 - eps) * tau
    sigma = -torch.log(one_minus.clamp_min(1e-6))
    sigma_prime = (1.0 - eps) / one_minus.clamp_min(1e-6)
    return sigma_prime / (torch.exp(sigma) - 1.0 + 1e-6)
