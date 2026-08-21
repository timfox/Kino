"""FiLM-conditioned U-Net exposure head (LatentHDR Sec. 4.1, Eq. 1)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.latenthdr.ev_embedding import ExposureValueEmbedding


class _FiLM(nn.Module):
    def __init__(self, cond_dim: int, channels: int) -> None:
        super().__init__()
        self.proj = nn.Linear(cond_dim, channels * 2)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        g, b = self.proj(cond).chunk(2, dim=-1)
        g = g.view(g.shape[0], -1, 1, 1)
        b = b.view(b.shape[0], -1, 1, 1)
        return x * (1.0 + g) + b


class _ResBlock(nn.Module):
    def __init__(self, ch: int, cond_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, ch), ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, ch), ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.film1 = _FiLM(cond_dim, ch)
        self.film2 = _FiLM(cond_dim, ch)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.film1(self.norm1(x), cond)))
        h = self.conv2(F.silu(self.film2(self.norm2(h), cond)))
        return x + h


class FiLMExposureUNet(nn.Module):
    """3-stage FiLM U-Net predicting exposure residual in VAE latent space."""

    def __init__(
        self,
        *,
        latent_channels: int = 16,
        base_ch: int = 32,
        cond_dim: int = 128,
        ev_bands: int = 32,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.ev_embed = ExposureValueEmbedding(embed_dim=ev_bands)
        self.ev_mlp = nn.Sequential(
            nn.Linear(ev_bands, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
        )
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4
        self.in_conv = nn.Conv2d(latent_channels, c1, 3, padding=1)
        self.down1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)
        self.mid = _ResBlock(c3, cond_dim)
        self.up2 = nn.ConvTranspose2d(c3, c2, 4, stride=2, padding=1)
        self.dec2 = _ResBlock(c2, cond_dim)
        self.up1 = nn.ConvTranspose2d(c2, c1, 4, stride=2, padding=1)
        self.dec1 = _ResBlock(c1, cond_dim)
        self.out_conv = nn.Conv2d(c1, latent_channels, 3, padding=1)

    def _cond(self, ev: Tensor, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        if ev.ndim == 0:
            ev = ev.reshape(1)
        emb = self.ev_embed(ev.to(device=device))
        return self.ev_mlp(emb.to(dtype=dtype))

    def forward(self, z_base: Tensor, ev: Tensor) -> Tensor:
        if z_base.dim() == 5:
            b, c, f, h, w = z_base.shape
            flat = z_base.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
            if ev.ndim == 0:
                ev_flat = ev.reshape(1).expand(b * f)
            elif ev.ndim == 1 and ev.shape[0] == b:
                ev_flat = ev.unsqueeze(1).expand(b, f).reshape(b * f)
            elif ev.numel() == 1:
                ev_flat = ev.reshape(1).expand(b * f)
            else:
                ev_flat = ev
            out = self._forward_2d(flat, ev_flat)
            return out.view(b, f, c, h, w).permute(0, 2, 1, 3, 4)
        return self._forward_2d(z_base, ev)

    def _forward_2d(self, z_base: Tensor, ev: Tensor) -> Tensor:
        b = z_base.shape[0]
        if ev.ndim == 0:
            ev = ev.reshape(1).expand(b)
        elif ev.ndim == 1 and ev.shape[0] == 1 and b > 1:
            ev = ev.expand(b)
        cond = self._cond(ev, device=z_base.device, dtype=z_base.dtype)
        h0 = F.silu(self.in_conv(z_base))
        h1 = F.silu(self.down1(h0))
        h2 = F.silu(self.down2(h1))
        h = self.mid(h2, cond)
        h = F.silu(self.up2(h) + h1)
        h = self.dec2(h, cond)
        h = F.silu(self.up1(h) + h0)
        h = self.dec1(h, cond)
        return z_base + self.out_conv(h)
