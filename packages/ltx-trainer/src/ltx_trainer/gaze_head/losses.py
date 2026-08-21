"""cVAE training objective (Liu et al., Eq. 1)."""

from __future__ import annotations

import torch
from torch import Tensor


def kl_divergence_gaussian(mu: Tensor, logvar: Tensor) -> Tensor:
    """KL(q(z|x) || N(0,I)) summed over latent dims, mean over batch."""
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(dim=-1).mean()


def cvae_loss(
    pred_head: Tensor,
    target_head: Tensor,
    mu: Tensor,
    logvar: Tensor,
    *,
    kl_weight: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    """L = ||ĥ - h||² + λ L_KL (Eq. 1)."""
    recon = torch.nn.functional.mse_loss(pred_head, target_head)
    kl = kl_divergence_gaussian(mu, logvar)
    total = recon + kl_weight * kl
    return total, {"recon": float(recon.detach()), "kl": float(kl.detach())}
