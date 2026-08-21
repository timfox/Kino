"""Panoramic Hybrid Attention: SWA + PSA (Eq. 12–13)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.panolm.psa import PanoramicSparseAttention


def sliding_window_attention(h: Tensor, window_size: int) -> Tensor:
    """Non-overlapping SWA (Eq. 4–6)."""
    b, l, d = h.shape
    if window_size >= l:
        return h
    pad = (window_size - l % window_size) % window_size
    if pad:
        h = F.pad(h, (0, 0, 0, pad))
    l_p = h.shape[1]
    w = window_size
    nwin = l_p // w
    chunks = h.view(b, nwin, w, d)
    outs = []
    for i in range(nwin):
        x = chunks[:, i]
        attn = F.softmax(torch.bmm(x, x.transpose(1, 2)) / (d**0.5), dim=-1)
        outs.append(torch.bmm(attn, x))
    out = torch.cat(outs, dim=1)[:, :l]
    return out


class PanoramicHybridBlock(nn.Module):
    def __init__(self, dim: int, window_size: int = 8, top_k: int = 32) -> None:
        super().__init__()
        self.psa = PanoramicSparseAttention(dim, top_k=top_k)
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, h: Tensor) -> Tensor:
        local = sliding_window_attention(h, self.window_size)
        global_ = self.psa(h)
        h = self.norm1(h + local + global_)
        h = self.norm2(h + self.ffn(h))
        return h
