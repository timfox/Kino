"""Geometry-free sequence vs channel conditioning (Sec. 3.2)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def patchify_tokens(feat: Tensor, patch_size: int = 2) -> Tensor:
    """B×C×H×W → B×N×C*ph*pw flattened per patch."""
    b, c, h, w = feat.shape
    ph, pw = h // patch_size, w // patch_size
    x = feat.reshape(b, c, ph, patch_size, pw, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5).reshape(b, ph * pw, -1)
    return x


def unpatchify_tokens(tokens: Tensor, h: int, w: int, patch_size: int, channels: int) -> Tensor:
    ph, pw = h // patch_size, w // patch_size
    b, n, _ = tokens.shape
    x = tokens.reshape(b, ph, pw, channels, patch_size, patch_size)
    x = x.permute(0, 3, 1, 4, 2, 5).reshape(b, channels, h, w)
    return x


def sequence_concat_conditioning(
    latent_pers: Tensor,
    latent_equi_noisy: Tensor,
    *,
    patch_size: int = 2,
) -> Tensor:
    """Concat([x_pers, y_t_equi]) along sequence dim for DiT (Sec. 3.2)."""
    t_p = patchify_tokens(latent_pers, patch_size)
    t_e = patchify_tokens(latent_equi_noisy, patch_size)
    return torch.cat([t_p, t_e], dim=1)


def channel_concat_conditioning(latent_pers: Tensor, latent_equi_noisy: Tensor) -> Tensor:
    """Baseline: pixel-aligned ERP projection + channel concat (Sec. 3.2 critique)."""
    if latent_pers.shape[-2:] != latent_equi_noisy.shape[-2:]:
        latent_pers = torch.nn.functional.interpolate(
            latent_pers, size=latent_equi_noisy.shape[-2:], mode="bilinear", align_corners=False
        )
    return torch.cat([latent_pers, latent_equi_noisy], dim=1)


class DiTBlockStub(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        h, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + h
        return x + self.ff(self.norm2(x))


class GeometryFreeDiTStub(nn.Module):
    """Minimal DiT denoiser on concatenated perspective + ERP tokens."""

    def __init__(self, token_dim: int, depth: int = 2, num_heads: int = 4) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([DiTBlockStub(token_dim, num_heads) for _ in range(depth)])
        self.out = nn.Linear(token_dim, token_dim)

    def forward(self, tokens: Tensor, n_pers_tokens: int) -> Tensor:
        x = tokens
        for blk in self.blocks:
            x = blk(x)
        x = self.out(x)
        return x[:, n_pers_tokens:, :]  # predict noise on ERP tokens only
