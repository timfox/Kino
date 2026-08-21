"""Learned JPEG color transform F and F^{-1} (Eq. 2–3, Fig. 3)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.seaotter.config import SeaotterConfig


def softsign(x: Tensor) -> Tensor:
    return x / (1.0 + x.abs())


def softsign_inverse(y: Tensor, scale: Tensor, eps: float = 1e-6) -> Tensor:
    """Algebraic inverse of y = scale * softsign(x) per channel."""
    s = scale.clamp(min=eps).view(1, -1, 1, 1)
    y_norm = (y / s).clamp(-0.999, 0.999)
    return y_norm / (1.0 - y_norm.abs())


class LearnedColorTransform(nn.Module):
    """F = A_α,β ∘ σ_s ∘ ConvW; identity-initialized per paper §2."""

    def __init__(self, cfg: SeaotterConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or SeaotterConfig()
        self.cfg = cfg
        self.conv_w = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=True)
        self.conv_fw = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=True)
        self.softsign_scale = nn.Parameter(torch.ones(3))
        self.affine_scale = nn.Parameter(torch.ones(3))
        self.affine_offset = nn.Parameter(torch.zeros(3))
        self._init_identity()

    def _init_identity(self) -> None:
        nn.init.dirac_(self.conv_w.weight)
        nn.init.zeros_(self.conv_w.bias)
        nn.init.dirac_(self.conv_fw.weight)
        nn.init.zeros_(self.conv_fw.bias)

    def forward(self, x: Tensor, *, training_noise: bool = False) -> Tensor:
        """x in [-1, 1] RGB → uint8-like [0, 255] learned space."""
        h = self.conv_w(x)
        h = self.softsign_scale.view(1, 3, 1, 1) * softsign(h)
        h = h * self.affine_scale.view(1, 3, 1, 1) + self.affine_offset.view(1, 3, 1, 1)
        if training_noise:
            h = h + torch.empty_like(h).uniform_(-0.5, 0.5)
        return h.clamp(0.0, 255.0)

    def inverse(self, y: Tensor) -> Tensor:
        """JPEG-decoded coefficients → display RGB in [-1, 1]."""
        off = self.affine_offset.view(1, -1, 1, 1)
        sc = self.affine_scale.view(1, -1, 1, 1).clamp(min=1e-6)
        h = (y - off) / sc
        h = softsign_inverse(h, self.softsign_scale)
        h = self.conv_fw(h)
        return h.clamp(-1.0, 1.0)
