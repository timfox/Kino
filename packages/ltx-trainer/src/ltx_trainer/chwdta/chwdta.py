"""ChWDTB / ChWDTA blocks (§III-A, Fig. 3)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.chwdta.lifting import LearnableLifting1D, iwtc, wtc


class ChWDMSA(nn.Module):
    """Windowed spatial MSA on wavelet-reparameterized channels (Eq. 1 steps 2–3)."""

    def __init__(self, dim: int, num_heads: int = 8, window: int = 8) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window = window
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        pad_h = (self.window - h % self.window) % self.window
        pad_w = (self.window - w % self.window) % self.window
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, hp, wp = x.shape
        out = torch.zeros_like(x)
        for i in range(0, hp, self.window):
            for j in range(0, wp, self.window):
                win = x[:, :, i : i + self.window, j : j + self.window]
                wb, wc, wh, ww = win.shape
                tokens = win.flatten(2).transpose(1, 2)
                qkv = self.qkv(tokens).reshape(b, wh * ww, 3, self.num_heads, -1)
                q, k, v = qkv.unbind(2)
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)
                attn = F.softmax(q @ k.transpose(-2, -1) / (wc // self.num_heads) ** 0.5, dim=-1)
                y = (attn @ v).transpose(1, 2).reshape(b, wh * ww, wc)
                y = self.proj(y).transpose(1, 2).reshape(wb, wc, wh, ww)
                out[:, :, i : i + self.window, j : j + self.window] = y
        return out[:, :, :h, :w]


class ChWDTB(nn.Module):
    """Channel-wise Wavelet-Domain Transformer Block."""

    def __init__(self, channels: int, num_heads: int = 8) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.lift = LearnableLifting1D(channels)
        self.attn = ChWDMSA(channels, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        residual = x
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)
        t = wtc(x, self.lift)
        t = self.attn(t)
        x = iwtc(t, self.lift)
        x = x + residual
        # FFN on spatial tokens
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = tokens + self.ff(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(b, c, h, w)
