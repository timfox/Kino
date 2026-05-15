"""FiLM-conditioned residual on VAE latents: ``z_e = z_base + f_θ(z_base, φ(e))``."""

from __future__ import annotations

import torch
import torch.nn as nn

from ltx_trainer.latenthdr.ev_embedding import EVConditionMLP


class FiLMSpatial3d(nn.Module):
    """Per-channel affine modulation from a global cond vector."""

    def __init__(self, channels: int, cond_dim: int) -> None:
        super().__init__()
        self.to_gamma_beta = nn.Linear(cond_dim, 2 * channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gb = self.to_gamma_beta(cond)
        g, b = gb.chunk(2, dim=-1)
        g = g.view(-1, x.shape[1], 1, 1, 1)
        b = b.view(-1, x.shape[1], 1, 1, 1)
        return x * (1.0 + torch.tanh(g)) + b


class FiLMResidualExposureHead(nn.Module):
    """Residual exposure head on latent tensors ``[B, C, F, H, W]``.

    Forward returns ``z_base + residual_scale * r`` where ``r`` is the stack of FiLM Conv3d blocks.
    """

    def __init__(
        self,
        latent_channels: int = 128,
        cond_dim: int = 128,
        num_bands: int = 32,
        num_layers: int = 4,
        residual_scale: float = 0.05,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.residual_scale = residual_scale
        self.ev_mlp = EVConditionMLP(num_bands=num_bands, hidden=256, out_dim=cond_dim)
        layers: list[nn.Module] = []
        for _ in range(num_layers):
            layers.append(
                nn.Conv3d(latent_channels, latent_channels, kernel_size=3, padding=1, bias=True),
            )
            layers.append(nn.SiLU())
            layers.append(FiLMSpatial3d(latent_channels, cond_dim))
        self.body = nn.ModuleList(layers)

    def forward(self, z_base: torch.Tensor, ev: torch.Tensor) -> torch.Tensor:
        """``z_base``: ``[B,C,F,H,W]``; ``ev``: ``[B]`` scalar EV in log₂ units."""
        cond = self.ev_mlp(ev)
        h = z_base
        i = 0
        while i < len(self.body):
            conv = self.body[i]
            act = self.body[i + 1]
            film = self.body[i + 2]
            h = film(act(conv(h)), cond)
            i += 3
        return z_base + self.residual_scale * h
