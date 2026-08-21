"""Channel-wise tokenization — per-electrode tokens."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ChannelWiseTokenizer(nn.Module):
    """Each electrode → patch tokens + shared CLS (Fig. 1)."""

    def __init__(self, num_channels: int, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.patch_embed = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos = nn.Parameter(torch.zeros(1, embed_dim, 32))
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, L)
        b, c, _ = x.shape
        tokens = []
        for ch in range(c):
            h = self.patch_embed(x[:, ch : ch + 1, :])
            h = h + self.pos[:, :, : h.shape[-1]]
            tokens.append(h.transpose(1, 2))
        stacked = torch.cat(tokens, dim=1)
        cls = self.cls.expand(b, -1, -1)
        return torch.cat([cls, stacked], dim=1)
