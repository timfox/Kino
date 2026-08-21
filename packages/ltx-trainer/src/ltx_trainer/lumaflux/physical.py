"""Physical feature extraction T_phys (Sec. 4.1, Eq. 6)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.lumaflux.color import bt2020_luma, pq_eotf, saturation


def luminance_gradient(y: Tensor) -> Tensor:
    gx = F.pad(y[:, :, :, 1:] - y[:, :, :, :-1], (0, 1))
    gy = F.pad(y[:, :, 1:, :] - y[:, :, :-1, :], (0, 0, 0, 1))
    return torch.sqrt(gx**2 + gy**2 + 1e-8)


def global_luminance_stats(y: Tensor) -> Tensor:
    """sg = [μY, σY, p95, p99] → flattened for MLP."""
    flat = y.reshape(y.shape[0], -1)
    mu = flat.mean(dim=1)
    sigma = flat.std(dim=1)
    p95 = torch.quantile(flat, 0.95, dim=1)
    p99 = torch.quantile(flat, 0.99, dim=1)
    return torch.stack([mu, sigma, p95, p99], dim=1)


def spectral_bands(y: Tensor, k: int = 8) -> Tensor:
    """FFT energy pooling into K bands (Sec. 4.1)."""
    if y.dim() == 3:
        y = y.unsqueeze(0)
    spec = torch.fft.rfft2(y, norm="ortho")
    mag = spec.abs()
    _, _, h, w = mag.shape
    band_edges = torch.linspace(0, h * w, k + 1, device=y.device).long()
    flat = mag.reshape(mag.shape[0], -1)
    bands = []
    for i in range(k):
        lo, hi = band_edges[i].item(), band_edges[i + 1].item()
        bands.append(flat[:, lo:hi].mean(dim=1))
    return torch.stack(bands, dim=1)


class PhysicalFeatureExtractor(nn.Module):
    """T_phys = Conv3×3([Y, log(1+|∇Y|), sat]) and global g = MLP(sg)."""

    def __init__(self, out_channels: int = 32) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, out_channels, 3, padding=1)
        self.global_mlp = nn.Sequential(
            nn.Linear(4, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, sdr: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            sdr: PQ-encoded SDR [B,3,H,W] in [0,1]
        Returns:
            t_phys [B,C,H,W], g_global [B,C], spectral r [B,K]
        """
        lin = pq_eotf(sdr)
        y = bt2020_luma(lin / lin.max().clamp(min=1e-3))
        grad = luminance_gradient(y)
        sat = saturation(sdr)
        stack = torch.cat([y, torch.log1p(grad), sat], dim=1)
        t_phys = self.conv(stack)
        sg = global_luminance_stats(y)
        g = self.global_mlp(sg)
        r = spectral_bands(y, k=8)
        return t_phys, g, r
