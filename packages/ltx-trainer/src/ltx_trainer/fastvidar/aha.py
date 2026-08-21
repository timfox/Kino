"""Alternative Hierarchical Attention — window / frame / global (Sec. III-B)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.fastvidar.config import FastViDARConfig


def _window_partition(feat: Tensor, wh: int, ww: int) -> Tensor:
    """[B*S, C, H, W] → [B*S, M, P, C] with P = wh*ww."""
    _, c, h, w = feat.shape
    h_pad = math.ceil(h / wh) * wh
    w_pad = math.ceil(w / ww) * ww
    if h_pad != h or w_pad != w:
        feat = F.pad(feat, (0, w_pad - w, 0, h_pad - h))
    feat = feat.view(feat.shape[0], c, h_pad // wh, wh, w_pad // ww, ww)
    feat = feat.permute(0, 2, 4, 3, 5, 1).contiguous()
    m = (h_pad // wh) * (w_pad // ww)
    return feat.view(feat.shape[0], m, wh * ww, c)


def _window_unpartition(tokens: Tensor, h: int, w: int, wh: int, ww: int) -> Tensor:
    """[B*S, M, P, C] → [B*S, C, H, W]."""
    bs, m, p, c = tokens.shape
    h_pad = math.ceil(h / wh) * wh
    w_pad = math.ceil(w / ww) * ww
    nh, nw = h_pad // wh, w_pad // ww
    x = tokens.view(bs, nh, nw, wh, ww, c).permute(0, 5, 1, 3, 2, 4).contiguous()
    x = x.view(bs, c, h_pad, w_pad)
    return x[:, :, :h, :w]


class AHABlock(nn.Module):
    """One AHA block: window → frame summary → global summary → broadcast."""

    def __init__(self, dim: int, num_heads: int = 4, use_global: bool = True) -> None:
        super().__init__()
        self.use_global = use_global
        self.norm1 = nn.LayerNorm(dim)
        self.win_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.frame_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.global_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.frame_embed = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.frame_embed, std=0.02)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.norm3 = nn.LayerNorm(dim)

    def forward(
        self,
        tokens: Tensor,
        *,
        num_frames: int,
    ) -> Tensor:
        """
        tokens: [B*S, M, P, C]
        """
        bs, m, p, c = tokens.shape
        s = num_frames
        b = bs // s

        # 1) Window attention (per window)
        flat = tokens.reshape(bs * m, p, c)
        win_out, _ = self.win_attn(self.norm1(flat), self.norm1(flat), self.norm1(flat))
        tokens = tokens + win_out.view(bs, m, p, c)

        # 2) Frame attention on summary tokens
        summary = tokens.mean(dim=2)  # [B*S, M, C]
        summary = summary + self.frame_embed
        frame_out, _ = self.frame_attn(self.norm2(summary), self.norm2(summary), self.norm2(summary))
        summary = summary + frame_out

        # 3) Global attention across frames
        if self.use_global:
            glob_in = summary.view(b, s * m, c)
            glob_out, _ = self.global_attn(
                self.norm2(glob_in), self.norm2(glob_in), self.norm2(glob_in)
            )
            summary = glob_out.reshape(bs, m, c)
        else:
            summary = frame_out

        # Broadcast summary to local tokens
        tokens = tokens + summary.unsqueeze(2)
        tokens = tokens + self.mlp(self.norm3(tokens))
        return tokens


class AHAStack(nn.Module):
    def __init__(self, cfg: FastViDARConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList(
            [
                AHABlock(cfg.hidden_dim, use_global=cfg.use_global_attention)
                for _ in range(cfg.num_aha_blocks)
            ]
        )

    def forward(self, feat: Tensor) -> Tensor:
        """feat: [B, S, C, H, W]"""
        b, s, c, h, w = feat.shape
        x = feat.reshape(b * s, c, h, w)
        wh, ww = self.cfg.window_h, self.cfg.window_w
        tokens = _window_partition(x, wh, ww)
        for block in self.blocks:
            tokens = block(tokens, num_frames=s)
        x = _window_unpartition(tokens, h, w, wh, ww)
        return x.view(b, s, c, h, w)
