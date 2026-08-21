"""Inference-time Blend / Repel steering (Eq. 16–18)."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.safedig.sae import SparseAutoencoder


def blend_steering(a: Tensor, a_tgt: Tensor, *, beta: float) -> Tensor:
    """Blend: a' = (1-β)a + βâ_tgt — Eq. (16)."""
    beta = max(0.0, min(beta, 1.0))
    return (1.0 - beta) * a + beta * a_tgt


def repel_steering(
    a: Tensor,
    sae: SparseAutoencoder,
    *,
    harm_mask: Tensor,
    gamma: float,
) -> Tensor:
    """Repel: a' = a - γ(â_harm - a) — Eq. (17–18)."""
    z = sae.encode(a)
    z_harm = z * harm_mask
    a_harm = sae.decode(z_harm)
    return a - gamma * (a_harm - a)


def steer_activation(
    a: Tensor,
    sae: SparseAutoencoder,
    a_tgt: Tensor,
    *,
    operator: str = "Blend",
    strength: float = 0.2,
    harm_mask: Tensor | None = None,
) -> Tensor:
    """Apply routed operator at inference."""
    if operator.lower() == "repel":
        if harm_mask is None:
            harm_mask = (sae.encode(a).abs() > sae.encode(a).abs().median()).float()
        return repel_steering(a, sae, harm_mask=harm_mask, gamma=strength)
    return blend_steering(a, a_tgt, beta=strength)


def estimate_harmful_mask(
    sae: SparseAutoencoder,
    harmful_activations: Tensor,
    *,
    quantile: float = 0.75,
) -> Tensor:
    """M_harm sparse mask from source/target safety features — Eq. (17)."""
    with torch.no_grad():
        z = sae.encode(harmful_activations)
        thresh = torch.quantile(z.abs(), quantile)
        mask = (z.abs() >= thresh).float().mean(dim=0)
    return (mask > 0.1).float()
