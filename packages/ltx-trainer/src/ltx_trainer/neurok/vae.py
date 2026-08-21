"""Conditional VAE for NEUROK learning (Sec. 4, Eq. 1)."""

from __future__ import annotations

import torch
from torch import Tensor


def kl_gaussian(
    mu_q: Tensor,
    logvar_q: Tensor,
    *,
    mu_p: Tensor | None = None,
) -> Tensor:
    """KL(q||p) for diagonal Gaussians; prior N(μ_p, I) if μ_p given else N(0,I)."""
    if mu_p is None:
        mu_p = torch.zeros_like(mu_q)
    var_q = logvar_q.exp()
    kl = 0.5 * (var_q + (mu_q - mu_p).pow(2) - 1.0 - logvar_q)
    return kl.sum(dim=-1).mean()


def conditional_vae_loss(
    delta_pred: Tensor,
    delta_target: Tensor,
    mu_q: Tensor,
    logvar_q: Tensor,
    *,
    mu_prior: Tensor,
    kl_weight: float = 0.01,
) -> dict[str, Tensor]:
    """L = ||δ_sample − δ_pred||² + λ D_KL(q||p) (Eq. 1)."""
    recon = torch.nn.functional.mse_loss(delta_pred, delta_target)
    kl = kl_gaussian(mu_q, logvar_q, mu_p=mu_prior)
    total = recon + kl_weight * kl
    return {"total": total, "recon": recon, "kl": kl}


def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    std = (0.5 * logvar).exp()
    eps = torch.randn_like(std)
    return mu + eps * std


def toy_encoder_decoder_step(
    *,
    num_samples: int,
    latent_tokens: int,
    token_dim: int,
    deform_dim: int,
    kl_weight: float = 0.01,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Differentiable smoke: linear E_cond / E_VAE / D proxies."""
    device = device or torch.device("cpu")
    torch.manual_seed(0)
    k = latent_tokens * token_dim
    n = num_samples * deform_dim

    e_cond = torch.randn(k, k, device=device) * 0.01
    e_vae = torch.randn(n + k, 2 * k, device=device) * 0.01
    decoder = torch.randn(n, k, device=device) * 0.01

    cond_feat = torch.randn(1, k, device=device)
    mu_prior = cond_feat @ e_cond
    delta_tgt = torch.randn(1, n, device=device)

    enc_in = torch.cat([delta_tgt, cond_feat], dim=-1)
    h = enc_in @ e_vae
    mu_q, logvar_q = h[..., :k], h[..., k : 2 * k].clamp(-4, 4)
    z = reparameterize(mu_q, logvar_q)
    delta_pred = z @ decoder.T

    losses = conditional_vae_loss(
        delta_pred, delta_tgt, mu_q, logvar_q, mu_prior=mu_prior, kl_weight=kl_weight
    )
    return {k: round(float(v.detach()), 4) for k, v in losses.items()}
