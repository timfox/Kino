"""Linear degradations y = Ax + n (Sec. 2, arXiv:2605.24769)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def apply_identity(x: Tensor) -> Tensor:
    return x


def apply_gaussian_blur(x: Tensor, *, kernel_size: int = 5, sigma: float = 1.0) -> Tensor:
    """Depthwise Gaussian blur per spectral band."""
    if x.dim() != 4:
        raise ValueError("x must be [B,C,H,W]")
    b, c, h, w = x.shape
    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - kernel_size // 2
    g1 = torch.exp(-0.5 * (coords / sigma) ** 2)
    g1 = g1 / g1.sum()
    g2 = g1[:, None] * g1[None, :]
    weight = g2.view(1, 1, kernel_size, kernel_size).expand(c, 1, kernel_size, kernel_size)
    pad = kernel_size // 2
    return F.conv2d(x, weight, padding=pad, groups=c)


def apply_superres_downsample(x: Tensor, *, scale: int = 4) -> Tensor:
    """Gaussian low-pass then uniform subsampling (paper SISR protocol)."""
    blurred = apply_gaussian_blur(x, kernel_size=5, sigma=1.0)
    return blurred[:, :, ::scale, ::scale]


def add_awgn(x: Tensor, sigma: float, *, generator: torch.Generator | None = None) -> Tensor:
    noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator) * sigma
    return (x + noise).clamp(0.0, 1.0)


def degrade(
    x: Tensor,
    task: str,
    *,
    noise_sigma: float = 0.1,
    scale: int = 4,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, str]:
    """Return ``(y, operator_name)`` for ``denoise`` | ``deblur`` | ``superres_x4``."""
    task = task.lower()
    if task in ("denoise", "denoising", "identity"):
        y = apply_identity(x)
    elif task in ("deblur", "deblurring", "blur"):
        y = apply_gaussian_blur(x)
    elif task in ("superres", "superres_x4", "sisr", "sr"):
        y = apply_superres_downsample(x, scale=scale)
    else:
        raise ValueError(f"Unknown task {task!r}")
    if noise_sigma > 0:
        y = add_awgn(y, noise_sigma, generator=generator)
    return y, task
