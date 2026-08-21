"""FRAPPE sensor encoder / cloud decoder stubs (§2, frozen G_A)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.seaotter.config import SeaotterConfig


class FrappeEncoder(nn.Module):
    """Lightweight patch projection → int8 latent prefix (variable rate n)."""

    def __init__(self, cfg: SeaotterConfig | None = None, latent_dim: int = 15) -> None:
        super().__init__()
        cfg = cfg or SeaotterConfig()
        self.cfg = cfg
        self.latent_dim = latent_dim
        self.proj = nn.Conv2d(3, latent_dim, kernel_size=4, stride=4)

    def forward(self, x: Tensor, *, n_channels: int = 12) -> Tensor:
        z = self.proj(x)
        n_channels = min(n_channels, z.shape[1])
        return z[:, :n_channels].round().clamp(-127, 127)


class FrappeDecoder(nn.Module):
    """Cloud synthesis G_S (~57M param scale stub for smoke)."""

    def __init__(self, cfg: SeaotterConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or SeaotterConfig()
        self.cfg = cfg
        self.up = nn.Sequential(
            nn.ConvTranspose2d(12, 64, 4, 2, 1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.GELU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: Tensor) -> Tensor:
        if z.shape[1] < 12:
            pad = torch.zeros(z.shape[0], 12 - z.shape[1], z.shape[2], z.shape[3], device=z.device)
            z = torch.cat([z, pad], dim=1)
        return self.up(z[:, :12])


def sensor_encode_throughput_mpx_s(cfg: SeaotterConfig | None = None, n: int = 12) -> float:
    """Table 1 encode MPx/s for FRAPPE / SEAOTTER (frozen encoder)."""
    cfg = cfg or SeaotterConfig()
    # Paper Table 1 cls @ n=12
    anchors = {3: 601.47, 6: 317.23, 9: 271.81, 12: 177.76, 15: 108.17}
    return anchors.get(n, 177.76)


def frappe_pipeline_smoke(cfg: SeaotterConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SeaotterConfig()
    torch.manual_seed(3940)
    enc = FrappeEncoder(cfg)
    dec = FrappeDecoder(cfg)
    x = torch.randn(2, 3, 128, 128)
    z = enc(x, n_channels=cfg.matched_frappe_n)
    y = dec(z.float())
    return {
        "latent_shape": list(z.shape),
        "recon_shape": list(y.shape),
        "encode_mpx_s_n12": sensor_encode_throughput_mpx_s(cfg, 12),
    }
