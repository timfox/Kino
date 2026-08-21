"""Three-stage inverse rendering smokes (Sec. 5)."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ltx_trainer.rgb_nir_ir.brdf import cross_spectral_roughness_metallic, mix_basis_brdf, rgb_edge_albedo_regularizer
from ltx_trainer.rgb_nir_ir.capture import isolate_nir_flash


def stage1_geometry_init_loss(rgb_pred: Tensor, rgb_gt: Tensor) -> Tensor:
    """Stage 1: 2DGS RGB photometric loss (L1 stub)."""
    return (rgb_pred - rgb_gt).abs().mean()


def stage2_nir_flash_loss(
    nir_pred: Tensor,
    nir_gt: Tensor,
    *,
    lambda_geom: float = 0.1,
    depth_normal_consistency: Tensor | None = None,
) -> dict[str, Tensor]:
    """Stage 2: Eq. 5 — L_NIR = L_rec + λ_geom L_geom + ... (subset)."""
    l_rec = (nir_pred - nir_gt).pow(2).mean()
    l_geom = depth_normal_consistency if depth_normal_consistency is not None else torch.tensor(0.0)
    total = l_rec + lambda_geom * l_geom
    return {"total": total, "l_rec": l_rec, "l_geom": l_geom}


def stage2_basis_optimization_smoke(
    nir_flash: Tensor,
    *,
    num_bases: int = 4,
    steps: int = 5,
) -> dict[str, Any]:
    """Toy optimize basis weights against flash NIR patch."""
    h, w = nir_flash.shape[-2], nir_flash.shape[-1]
    device = nir_flash.device
    n_dot_i = torch.full((h, w), 0.8, device=device)
    n_dot_o = torch.full((h, w), 0.7, device=device)
    n_dot_h = torch.full((h, w), 0.85, device=device)

    logits = torch.randn(h, w, num_bases, device=device, requires_grad=True)
    basis_rho = torch.linspace(0.2, 0.9, num_bases, device=device).view(1, 1, -1).expand(h, w, -1)
    basis_sigma = torch.linspace(0.1, 0.5, num_bases, device=device).view(1, 1, -1).expand(h, w, -1)
    basis_m = torch.linspace(0.0, 0.8, num_bases, device=device).view(1, 1, -1).expand(h, w, -1)

    opt = torch.optim.Adam([logits], lr=0.05)
    target = nir_flash.detach()
    for _ in range(steps):
        opt.zero_grad()
        weights = torch.softmax(logits, dim=-1)
        pred, sigma, metallic = mix_basis_brdf(weights, basis_rho, basis_sigma, basis_m, n_dot_i, n_dot_o, n_dot_h)
        loss = (pred - target).pow(2).mean()
        loss.backward()
        opt.step()

    weights = torch.softmax(logits.detach(), dim=-1)
    sigma_out, m_out = cross_spectral_roughness_metallic(weights, basis_sigma, basis_m)
    return {
        "final_mse": float((pred.detach() - target).pow(2).mean().item()),
        "roughness_mean": float(sigma_out.mean().item()),
        "metallic_mean": float(m_out.mean().item()),
    }


def stage3_rgb_environment_loss(
    rgb_pred: Tensor,
    rgb_gt: Tensor,
    grad_rgb: Tensor,
    grad_nir: Tensor,
    *,
    lambda_edge: float = 0.01,
) -> dict[str, Tensor]:
    """Stage 3: Eq. 12 — L_RGB = L_rec + λ L_edge."""
    l_rec = (rgb_pred - rgb_gt).pow(2).mean()
    l_edge = rgb_edge_albedo_regularizer(grad_rgb, grad_nir).mean()
    return {"total": l_rec + lambda_edge * l_edge, "l_rec": l_rec, "l_edge": l_edge}


def monte_carlo_rgb_pixel_smoke(
    rho_rgb: Tensor,
    env_sample: Tensor,
    *,
    num_samples: int = 8,
) -> Tensor:
    """Eq. 10 stub: average env × diffuse albedo samples."""
    samples = env_sample.unsqueeze(0).expand(num_samples, -1)
    return (rho_rgb * samples.mean(dim=0)).mean()
