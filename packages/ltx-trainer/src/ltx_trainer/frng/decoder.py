"""Universal neural appearance decoder (paper Sec. 4.3, Eq. 13; supplementary Sec. 8.3)."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ltx_trainer.frng.config import FRNGConfig


class UniversalNeuralAppearanceDecoder(nn.Module):
    """``ρ(g, ω_o, ω_i) = Θ(concat(g, ω_o, ω_i))`` — small MLP shared across objects."""

    def __init__(self, cfg: FRNGConfig, *, mat_feat_dim: int) -> None:
        super().__init__()
        in_dim = mat_feat_dim + 3 + 3
        layers: list[nn.Module] = []
        w = cfg.appearance_decoder_width
        d = in_dim
        for i in range(cfg.appearance_decoder_layers - 1):
            layers.extend([nn.Linear(d, w), nn.GELU()])
            d = w
        layers.append(nn.Linear(d, cfg.appearance_out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, g_mat: Tensor, omega_o: Tensor, omega_i: Tensor) -> Tensor:
        """``g_mat`` (B, R, F), directions (B, R, 3) → radiance (B, R, out_dim)."""
        x = torch.cat([g_mat, omega_o, omega_i], dim=-1)
        return self.net(x)
