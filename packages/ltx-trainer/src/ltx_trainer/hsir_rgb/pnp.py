"""Half-quadratic splitting (HQS) stub for plug-and-play restoration (DPIR-style, arXiv:2605.24769)."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from ltx_trainer.hsir_rgb.operators import apply_gaussian_blur, apply_superres_downsample


def hqs_denoise_step(x: Tensor, y: Tensor, z: Tensor, *, mu: float) -> Tensor:
    """Data step for A = I: closed form ``(y + μ z) / (1 + μ)``."""
    return (y + mu * z) / (1.0 + mu)


def hqs_blur_adjoint_step(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    *,
    mu: float,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> Tensor:
    """One blur proximal-gradient style update (small-step stub, not full deconvolution)."""
    blurred = apply_gaussian_blur(x, kernel_size=kernel_size, sigma=sigma)
    resid = blurred - y
    grad = apply_gaussian_blur(resid, kernel_size=kernel_size, sigma=sigma)
    return x - (1.0 / (1.0 + mu)) * grad + (mu / (1.0 + mu)) * z


def hqs_superres_upsample_step(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    *,
    mu: float,
    scale: int = 4,
) -> Tensor:
    """Upsample LR observation and fuse with denoised prior (nearest stub)."""
    b, c, h_lr, w_lr = y.shape
    y_up = torch.nn.functional.interpolate(y, scale_factor=scale, mode="nearest")
    if y_up.shape[-2:] != z.shape[-2:]:
        y_up = torch.nn.functional.interpolate(y_up, size=z.shape[-2:], mode="bilinear", align_corners=False)
    return (y_up + mu * z) / (1.0 + mu)


def pnp_hqs_restore(
    y: Tensor,
    denoiser: Callable[[Tensor], Tensor],
    *,
    task: str = "denoise",
    num_iters: int = 3,
    mu: float = 1.0,
    scale: int = 4,
) -> Tensor:
    """Alternate denoiser proximal + simple data fidelity (HQS outer loop stub)."""
    x = y.clone()
    if task in ("superres", "superres_x4", "sisr", "sr"):
        x = torch.nn.functional.interpolate(y, scale_factor=scale, mode="bilinear", align_corners=False)
    for _ in range(num_iters):
        z = denoiser(x)
        if task in ("denoise", "denoising", "identity"):
            x = hqs_denoise_step(x, y, z, mu=mu)
        elif task in ("deblur", "deblurring", "blur"):
            x = hqs_blur_adjoint_step(x, y, z, mu=mu)
        elif task in ("superres", "superres_x4", "sisr", "sr"):
            x = hqs_superres_upsample_step(x, y, z, mu=mu, scale=scale)
        else:
            raise ValueError(f"Unknown task {task!r}")
    return x.clamp(0.0, 1.0)
