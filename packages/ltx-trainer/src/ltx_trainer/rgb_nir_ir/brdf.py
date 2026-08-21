"""Disney BRDF + RGB–NIR cross-spectral sharing (Eqs. 3–9)."""

from __future__ import annotations

import torch
from torch import Tensor


def ggx_distribution(n_dot_h: Tensor, roughness: Tensor, eps: float = 1e-6) -> Tensor:
    """Microfacet normal distribution D (GGX)."""
    alpha = roughness.clamp(min=eps) ** 2
    denom = torch.pi * ((n_dot_h**2) * (alpha**2 - 1.0) + 1.0).clamp(min=eps) ** 2
    return alpha**2 / denom


def fresnel_schlick(cos_theta: Tensor, f0: Tensor) -> Tensor:
    return f0 + (1.0 - f0) * (1.0 - cos_theta.clamp(0.0, 1.0)) ** 5


def smith_ggx_g1(n_dot_v: Tensor, roughness: Tensor, eps: float = 1e-6) -> Tensor:
    alpha = roughness.clamp(min=eps) ** 2
    tan2 = (1.0 - n_dot_v**2).clamp(min=0.0) / (n_dot_v**2).clamp(min=eps)
    return 2.0 / (1.0 + torch.sqrt(1.0 + alpha**2 * tan2))


def disney_brdf_nir(
    n_dot_i: Tensor,
    n_dot_o: Tensor,
    n_dot_h: Tensor,
    *,
    rho_nir: Tensor,
    roughness: Tensor,
    metallic: Tensor,
) -> Tensor:
    """Single-basis NIR Disney BRDF radiance factor (Eq. 4, simplified)."""
    diffuse = (1.0 - metallic) * rho_nir / torch.pi
    d = ggx_distribution(n_dot_h, roughness)
    f0 = metallic * 0.04 + (1.0 - metallic) * rho_nir
    f = fresnel_schlick(n_dot_o, f0)
    g = smith_ggx_g1(n_dot_i, roughness) * smith_ggx_g1(n_dot_o, roughness)
    spec = d * f * g / (4.0 * n_dot_i.clamp(min=1e-6) * n_dot_o.clamp(min=1e-6))
    return diffuse + spec


def mix_basis_brdf(
    weights: Tensor,
    basis_rho: Tensor,
    basis_sigma: Tensor,
    basis_m: Tensor,
    n_dot_i: Tensor,
    n_dot_o: Tensor,
    n_dot_h: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Weighted basis BRDFs (Eq. 3–4); returns radiance, sigma, m."""
    w = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    radiance = torch.zeros_like(n_dot_i)
    for k in range(weights.shape[-1]):
        radiance = radiance + w[..., k] * disney_brdf_nir(
            n_dot_i,
            n_dot_o,
            n_dot_h,
            rho_nir=basis_rho[..., k],
            roughness=basis_sigma[..., k],
            metallic=basis_m[..., k],
        )
    sigma = (w * basis_sigma).sum(dim=-1)
    metallic = (w * basis_m).sum(dim=-1)
    return radiance, sigma, metallic


def cross_spectral_roughness_metallic(
    weights: Tensor,
    basis_sigma: Tensor,
    basis_m: Tensor,
) -> tuple[Tensor, Tensor]:
    """Eq. 6: share σ and m to RGB stage."""
    w = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    sigma = (w * basis_sigma).sum(dim=-1)
    metallic = (w * basis_m).sum(dim=-1)
    return sigma, metallic


def rgb_edge_albedo_regularizer(
    grad_rgb_albedo: Tensor,
    grad_nir_albedo: Tensor,
    k: float = 1.0,
) -> Tensor:
    """Eq. 13: exp(-k|∇ρ_NIR|) |∇ρ_RGB|."""
    return torch.exp(-k * grad_nir_albedo.abs()) * grad_rgb_albedo.abs()
