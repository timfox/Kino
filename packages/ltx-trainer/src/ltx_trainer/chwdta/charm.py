"""ChARM slice-based entropy model stub (§III-B, Fig. 4)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.chwdta.chwdta import ChWDTB


class ChARMEntropy(nn.Module):
    """Sequential slice Gaussian likelihood with ChWDTB AttentionNet."""

    def __init__(self, slice_channels: int, hyper_channels: int = 16) -> None:
        super().__init__()
        ctx_c = slice_channels + hyper_channels
        self.attn_net = ChWDTB(ctx_c)
        self.param_net = nn.Sequential(
            nn.Conv2d(ctx_c, slice_channels, 1),
            nn.GELU(),
            nn.Conv2d(slice_channels, slice_channels * 2, 1),
        )
        self.slice_channels = slice_channels

    def forward(
        self,
        y_prior: Tensor,
        z_hat: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return (mean, scale) for discretized Gaussian (Eq. 15)."""
        ctx = torch.cat([y_prior, z_hat], dim=1)
        ctx = self.attn_net(ctx)
        params = self.param_net(ctx)
        mean, scale = params.chunk(2, dim=1)
        scale = torch.exp(scale).clamp(min=1e-4)
        return mean, scale


def charm_smoke(num_slices: int = 8) -> dict[str, Any]:
    sc, zc = 16, 16
    m = ChARMEntropy(sc, hyper_channels=zc)
    z = torch.randn(1, zc, 8, 8)
    y_prior = torch.zeros(1, sc, 8, 8)
    shapes = []
    for _ in range(num_slices):
        mu, sig = m(y_prior, z)
        shapes.append({"mean": list(mu.shape), "scale": list(sig.shape)})
        y_prior = y_prior + torch.randn_like(y_prior) * 0.1
    return {"num_slices": num_slices, "slice_shapes": shapes}
