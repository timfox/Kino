"""Feed-forward image transform net (Johnson et al.) with conditional instance norm."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.anthropocam.config import AnthropoCamConfig


class ConditionalInstanceNorm2d(nn.Module):
    """Dumoulin-style affine parameters per style index."""

    def __init__(self, num_features: int, num_styles: int) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.gamma = nn.Embedding(num_styles, num_features)
        self.beta = nn.Embedding(num_styles, num_features)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, x: Tensor, style_id: Tensor) -> Tensor:
        y = self.norm(x)
        g = self.gamma(style_id).unsqueeze(-1).unsqueeze(-1)
        b = self.beta(style_id).unsqueeze(-1).unsqueeze(-1)
        return g * y + b


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, num_styles: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.cin1 = ConditionalInstanceNorm2d(channels, num_styles)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.cin2 = ConditionalInstanceNorm2d(channels, num_styles)

    def forward(self, x: Tensor, style_id: Tensor) -> Tensor:
        y = F.relu(self.cin1(self.conv1(x), style_id))
        y = self.cin2(self.conv2(y), style_id)
        return x + y


class AnthropoTransformNet(nn.Module):
    """Single-pass stylization network for mobile deployment."""

    def __init__(self, cfg: AnthropoCamConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or AnthropoCamConfig()
        self.cfg = cfg
        n = cfg.num_styles
        c = 32
        self.down = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, c, 3),
            nn.ReLU(inplace=True),
        )
        self.res = nn.ModuleList([ResidualBlock(c, n) for _ in range(3)])
        self.up = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(c, 3, 3),
            nn.Sigmoid(),
        )

    def forward(self, content: Tensor, style_id: Tensor | None = None) -> Tensor:
        if style_id is None:
            style_id = torch.zeros(content.shape[0], dtype=torch.long, device=content.device)
        x = self.down(content)
        for block in self.res:
            x = block(x, style_id)
        x = self.up(x)
        return self.out(x)
