"""Smoke-scale Stage-II DiT block with Anchor-Guided Injector."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.pixelwizard.config import PixelWizardConfig
from ltx_trainer.pixelwizard.injector import AnchorGuidedInjector


class PixelWizardDiTBlock(nn.Module):
    """Single DiT block + anchor injection for smoke tests."""

    def __init__(self, channels: int, cfg: PixelWizardConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or PixelWizardConfig()
        self.cfg = cfg
        self.norm = nn.LayerNorm(channels)
        self.self_attn = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)
        self.injector = AnchorGuidedInjector(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(
        self,
        z: Tensor,
        anchor: Tensor,
        timestep: Tensor,
        step_size: Tensor,
    ) -> Tensor:
        b, c, h, w = z.shape
        tokens = z.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)
        attn_out, _ = self.self_attn(tokens, tokens, tokens)
        tokens = tokens + attn_out
        tokens = tokens + self.ff(tokens)
        z_mid = tokens.transpose(1, 2).view(b, c, h, w)
        return self.injector(z_mid, anchor, timestep, step_size)


class PixelWizardVelocityHead(nn.Module):
    """Predict shortcut velocity field s_θ(x_t, t, Δt)."""

    def __init__(self, in_channels: int = 16, hidden: int = 32) -> None:
        super().__init__()
        self.block = PixelWizardDiTBlock(hidden)
        self.in_proj = nn.Conv2d(in_channels, hidden, 3, padding=1)
        self.anchor_proj = nn.Conv2d(in_channels, hidden, 1)
        self.out_proj = nn.Conv2d(hidden, in_channels, 3, padding=1)

    def forward(
        self,
        x: Tensor,
        anchor: Tensor,
        t: Tensor,
        delta_t: Tensor,
    ) -> Tensor:
        h = self.in_proj(x)
        a = self.anchor_proj(anchor)
        h = self.block(h, a, t, delta_t)
        return self.out_proj(h)
