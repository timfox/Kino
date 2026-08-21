"""MaskAdapter: mask sequence → latent residual ΔZ (Sec. 4.2)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.comogen.config import CoMoGenConfig


def downsample_mask_sequence(mask: Tensor, cfg: CoMoGenConfig | None = None) -> Tensor:
    """Downsample mask to latent resolution: spatial /8, temporal OR over 4 frames."""
    cfg = cfg or CoMoGenConfig()
    # mask: [B, 1, T, H, W] or [T, H, W]
    if mask.dim() == 3:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 4:
        mask = mask.unsqueeze(1)
    b, _, t, h, w = mask.shape
    # spatial pool
    m = torch.nn.functional.max_pool2d(
        mask.reshape(b * t, 1, h, w),
        kernel_size=cfg.spatial_downsample,
        stride=cfg.spatial_downsample,
    )
    _, _, h8, w8 = m.shape
    m = m.reshape(b, t, h8, w8)
    # temporal OR windows of 4
    t_out = (t + cfg.temporal_downsample - 1) // cfg.temporal_downsample
    chunks = []
    for i in range(t_out):
        start = i * cfg.temporal_downsample
        end = min(start + cfg.temporal_downsample, t)
        chunks.append(m[:, start:end].amax(dim=1))
    return torch.stack(chunks, dim=1).unsqueeze(1)  # [B, 1, T/4, H/8, W/8]


class MaskAdapter(nn.Module):
    """Two Conv3D layers + linear projection (Sec. 4.2, Fig. 2)."""

    def __init__(self, cfg: CoMoGenConfig | None = None):
        super().__init__()
        cfg = cfg or CoMoGenConfig()
        c = cfg.latent_channels
        self.conv1 = nn.Conv3d(1, c // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(c // 2, c, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(c)
        self.proj = nn.Linear(c, c)

    def forward(self, mask_latent: Tensor) -> Tensor:
        """mask_latent: [B, 1, T', H', W'] -> ΔZ same shape as video latent [B, C, T', H', W']."""
        x = torch.relu(self.conv1(mask_latent))
        x = torch.relu(self.conv2(x))
        # channel-last norm + linear per spatial-temporal location
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(-1, c)
        x = self.proj(self.norm(x))
        x = x.reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3)
        return x


def mask_to_delta(
    mask: Tensor,
    adapter: MaskAdapter | None = None,
    cfg: CoMoGenConfig | None = None,
) -> Tensor:
    cfg = cfg or CoMoGenConfig()
    adapter = adapter or MaskAdapter(cfg)
    m = downsample_mask_sequence(mask, cfg)
    m = (m - m.mean()) / (m.std() + 1e-6)
    return adapter(m)
