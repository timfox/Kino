"""Stage I VAE temporal loss + Stage II fusion + latent refinement (Eq. 1–5)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.drfusion.config import DRFusionConfig


def latent_temporal_loss(
    z_prev: Tensor,
    z_curr: Tensor,
    flow: Tensor,
    mask: Tensor | None = None,
    eps: float = 1e-6,
) -> Tensor:
    """Eq. (1): occlusion-aware latent warping L_temp."""
    warped = warp_latent(z_prev, flow)
    diff = (warped - z_curr) ** 2
    if mask is not None:
        diff = diff * mask
        denom = mask.sum() + eps
    else:
        denom = diff.numel() / diff.shape[0] + eps
    return diff.sum() / denom


def warp_latent(z: Tensor, flow: Tensor) -> Tensor:
    """Bilinear warp z using flow (B, 2, H, W) at latent resolution."""
    b, _, h, w = z.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=z.device, dtype=z.dtype),
        torch.linspace(-1, 1, w, device=z.device, dtype=z.dtype),
        indexing="ij",
    )
    grid = torch.stack([xx + flow[:, 0] / max(w - 1, 1), yy + flow[:, 1] / max(h - 1, 1)], dim=-1)
    return F.grid_sample(z, grid, mode="bilinear", padding_mode="border", align_corners=True)


def fusion_pixel_loss(
    pred: Tensor,
    ir: Tensor,
    vis: Tensor,
    cfg: DRFusionConfig | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Eq. (4) proxy: L1 + gradient + SSIM-style terms."""
    cfg = cfg or DRFusionConfig()
    l_perc = F.l1_loss(pred, vis)
    l_ssim = 1.0 - _ssim_proxy(pred, vis)
    l_grad = F.l1_loss(_sobel_mag(pred), _sobel_mag(ir))
    l_int = F.l1_loss(pred, 0.5 * (ir + vis))
    total = (
        cfg.lambda_p * l_perc
        + cfg.lambda_s * l_ssim
        + cfg.lambda_g * l_grad
        + cfg.lambda_i * l_int
    )
    return total, {
        "perc": float(l_perc.detach()),
        "ssim": float(l_ssim.detach()),
        "grad": float(l_grad.detach()),
        "int": float(l_int.detach()),
    }


def latent_refinement_energy(
    z: Tensor,
    decode_fn,
    ir: Tensor,
    vis: Tensor,
    z_hat: Tensor,
    cfg: DRFusionConfig | None = None,
) -> Tensor:
    """Eq. (5) energy J(D(z), I_IR, I_VI) + λ_reg ||z − z_hat||²."""
    cfg = cfg or DRFusionConfig()
    decoded = decode_fn(z)
    align = F.l1_loss(decoded, ir) + F.l1_loss(decoded, vis)
    reg = F.mse_loss(z, z_hat)
    return align + cfg.lambda_reg * reg


def cooperative_latent_update(z_hat: Tensor, z_star: Tensor, gamma: float = 0.5) -> Tensor:
    """ẑ_new = (1 − γ) · ẑ + γ · z* (Sec. 3.2.1)."""
    return (1.0 - gamma) * z_hat + gamma * z_star


def _ssim_proxy(a: Tensor, b: Tensor) -> Tensor:
    c1, c2 = 0.01**2, 0.03**2
    mu_a = a.mean(dim=(-2, -1), keepdim=True)
    mu_b = b.mean(dim=(-2, -1), keepdim=True)
    sigma_a = ((a - mu_a) ** 2).mean(dim=(-2, -1), keepdim=True)
    sigma_b = ((b - mu_b) ** 2).mean(dim=(-2, -1), keepdim=True)
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean(dim=(-2, -1), keepdim=True)
    num = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (sigma_a + sigma_b + c2)
    return (num / den).mean()


def _sobel_mag(x: Tensor) -> Tensor:
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    ky = kx.transpose(-2, -1)
    c = x.shape[1]
    kx = kx.expand(c, 1, 3, 3)
    ky = ky.expand(c, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1, groups=c)
    gy = F.conv2d(x, ky, padding=1, groups=c)
    return torch.sqrt(gx**2 + gy**2 + 1e-6)
