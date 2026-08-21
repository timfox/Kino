"""GenEnvLighting: EV0 → underexposed panorama (Sec. 3.3, Flux Kontext + LoRA stub)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.lightharmony3d.config import GENENV_TARGET_EV


class GenEnvLighting(nn.Module):
    """Maps base EV0 panorama to radiometrically truncated exposure (e.g. EV−3)."""

    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, 3, 3, padding=1),
        )
        self.ev_scale = 2.0 ** (-GENENV_TARGET_EV)

    def forward(self, ev0: Tensor) -> Tensor:
        delta = self.net(ev0)
        under = (ev0 * (1.0 / self.ev_scale) + 0.1 * delta.tanh()).clamp(0, 1)
        return under
