"""SEANet encoder/decoder stub (Sec. 3.1.1)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.tqcodec.config import DOWNSAMPLE_FACTORS, TQCodecConfig


class SEANetBlock(nn.Module):
    def __init__(self, channels: int, stride: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=7, stride=stride, padding=3)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.conv(x))


class SEANetEncoder(nn.Module):
    """Lightweight SEANet-style encoder with LSTM (Sec. 3.1.1)."""

    def __init__(self, cfg: TQCodecConfig) -> None:
        super().__init__()
        c = cfg.encoder_dim
        self.in_conv = nn.Conv1d(1, c, kernel_size=7, padding=3)
        self.blocks = nn.ModuleList(SEANetBlock(c, s) for s in DOWNSAMPLE_FACTORS)
        self.lstm = nn.LSTM(c, c, batch_first=True)
        self.out = nn.Conv1d(c, cfg.latent_dim, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, 1, T)
        h = self.in_conv(x)
        for b in self.blocks:
            h = b(h)
        h = h.transpose(1, 2)
        h, _ = self.lstm(h)
        h = h.transpose(1, 2)
        return self.out(h)


class SEANetDecoder(nn.Module):
    """SEANet decoder — target ~6.31 GMACs (Sec. 3.1.1)."""

    def __init__(self, cfg: TQCodecConfig) -> None:
        super().__init__()
        c = cfg.decoder_dim
        self.in_conv = nn.Conv1d(cfg.latent_dim, c, kernel_size=1)
        self.lstm = nn.LSTM(c, c, batch_first=True)
        self.blocks = nn.ModuleList()
        for s in reversed(DOWNSAMPLE_FACTORS):
            self.blocks.append(
                nn.Sequential(
                    nn.ConvTranspose1d(c, c, kernel_size=s * 2, stride=s, padding=s // 2),
                    nn.GELU(),
                )
            )
        self.out = nn.Conv1d(c, 1, kernel_size=7, padding=3)

    def forward(self, z: Tensor) -> Tensor:
        h = self.in_conv(z)
        h = h.transpose(1, 2)
        h, _ = self.lstm(h)
        h = h.transpose(1, 2)
        for b in self.blocks:
            h = b(h)
        return torch.tanh(self.out(h))
