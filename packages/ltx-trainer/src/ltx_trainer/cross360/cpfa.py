"""Cross Projection Feature Alignment — Eq. (1), Fig. 2."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SwitchableNorm(nn.Module):
    """Lightweight switchable normalization stub (SN, Sec. III-C.3)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)
        self.ln = nn.GroupNorm(1, channels)
        self.weight = nn.Parameter(torch.ones(3))

    def forward(self, x: Tensor) -> Tensor:
        w = torch.softmax(self.weight, dim=0)
        return w[0] * self.bn(x) + w[1] * self.ln(x) + w[2] * x


class CrossProjectionFeatureAlignment(nn.Module):
    """MHSA over TP patches + cross-attention TP←ERP (Eq. 1)."""

    def __init__(self, channels: int, num_heads: int = 4, tp_in_channels: int = 3) -> None:
        super().__init__()
        self.tp_in = nn.Conv2d(tp_in_channels, channels, 1) if tp_in_channels != channels else nn.Identity()
        self.tp_self_attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm_tp = nn.LayerNorm(channels)
        self.norm_erp = SwitchableNorm(channels)
        self.wq = nn.Linear(channels, channels, bias=False)
        self.wk = nn.Linear(channels, channels, bias=False)
        self.wv = nn.Linear(channels, channels, bias=False)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, f_erp: Tensor, tp_patches: Tensor) -> Tensor:
        """
        f_erp: (B, C, H, W)
        tp_patches: (B, N, C, ph, pw)
        returns F_CA: (B, C, H, W) aligned in ERP layout
        """
        b, c, h, w = f_erp.shape
        _, n, cin, ph, pw = tp_patches.shape
        tp_feat = self.tp_in(tp_patches.reshape(b * n, cin, ph, pw)).reshape(b, n, c, ph, pw)

        tp_tokens = tp_feat.mean(dim=(-2, -1))  # (B, N, C)
        tp_ctx, _ = self.tp_self_attn(tp_tokens, tp_tokens, tp_tokens)
        tp_ctx = self.norm_tp(tp_ctx + tp_tokens)

        erp_n = self.norm_erp(f_erp)
        erp_tokens = erp_n.flatten(2).transpose(1, 2)  # (B, HW, C)

        aligned_patches: list[Tensor] = []
        for i in range(n):
            q = self.wq(tp_ctx[:, i : i + 1, :])  # (B, 1, C)
            k = self.wk(erp_tokens)
            v = self.wv(erp_tokens)
            attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / (c**0.5), dim=-1)
            out = torch.bmm(attn, v)  # (B, 1, C)
            aligned_patches.append(out)

        aligned = torch.cat(aligned_patches, dim=1)  # (B, N, C)
        # map patch summaries back to ERP grid (mean broadcast stub)
        f_ca = aligned.mean(dim=1).unsqueeze(-1).unsqueeze(-1).expand(b, c, h, w)
        return self.proj_out(f_ca + f_erp)
