"""DownConv, CDAM frequency path, WSB, RGD building blocks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.msfet_e2v.attention import CrossDomainAttention
from ltx_trainer.msfet_e2v.wavelet import haar_dwt2d, haar_idwt2d


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        h = F.silu(self.conv1(x))
        return x + self.conv2(h)


class DownConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.conv(x))


class CDAMBlock(nn.Module):
    """Spatio-temporal RB + Haar frequency branch + cross-domain attention."""

    def __init__(self, channels: int, embed_dim: int) -> None:
        super().__init__()
        self.spatial_rb = ResidualBlock(channels)
        self.ll_rb = ResidualBlock(channels)
        self.hf_rb = ResidualBlock(channels * 3)
        self.freq_reduce = nn.Conv2d(channels * 4, channels, 3, padding=1)
        self.embed = nn.Conv2d(channels, embed_dim, kernel_size=3, stride=1, padding=1)
        self.attn = CrossDomainAttention(embed_dim)
        self.out_proj = nn.Conv2d(embed_dim, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        st = self.spatial_rb(x)
        ll, lh, hl, hh = haar_dwt2d(x)
        ll_p = self.ll_rb(ll)
        hf = self.hf_rb(torch.cat([lh, hl, hh], dim=1))
        freq = self.freq_reduce(torch.cat([ll_p, hf], dim=1))
        b, _, h, w = st.shape
        q_map = self.embed(st)
        kv_map = self.embed(freq)
        q = q_map.flatten(2).transpose(1, 2)
        kv = kv_map.flatten(2).transpose(1, 2)
        tokens = self.attn(q, kv)
        out = tokens.transpose(1, 2).view(b, -1, h, w)
        return st + self.out_proj(out)


class WaveletSkipBlock(nn.Module):
    """WSB: spatial RB + selective HH processing + IWT (Fig. 4b)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.spatial_rb = ResidualBlock(channels)
        self.hh_rb = ResidualBlock(channels)

    def forward(self, x: Tensor) -> Tensor:
        spatial = self.spatial_rb(x)
        ll, lh, hl, hh = haar_dwt2d(x)
        hh_p = self.hh_rb(hh)
        freq = haar_idwt2d(ll, lh, hl, hh_p)
        return spatial + freq


class ResidualGuidedDecoder(nn.Module):
    """RGD: upsample ×2 + RB (+ optional channel align)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.rb = ResidualBlock(in_ch)
        self.align = nn.Conv2d(in_ch, out_ch, 3, padding=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.align(self.rb(self.up(x)))
