"""Event-Illumination Collaborative Interaction (EICI, Sec. 3.3)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class _CovarianceGather(nn.Module):
    """Forward gathering G(X, T): cross-covariance attention (XCiT-style)."""

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = max(dim // num_heads, 1)
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, auxiliary: Tensor, primary: Tensor) -> tuple[Tensor, Tensor]:
        """``auxiliary``, ``primary``: ``[B, HW, C]`` → updated primary + channel attention ``[B,C,C]``."""
        b, n, c = primary.shape
        q = self.q_proj(primary).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(auxiliary).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(auxiliary).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, n, c)
        primary_new = primary + self.out_proj(out)
        # Channel covariance proxy for backward injection (Eq. 5–6)
        a = torch.matmul(primary.transpose(1, 2), primary_new) / max(n, 1)
        a = a / (a.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        return primary_new, a


class _BackwardInject(nn.Module):
    """Backward injection I(T', A) + X."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, fused: Tensor, attn: Tensor, auxiliary: Tensor) -> Tensor:
        b, n, c = fused.shape
        # Map fused tokens through stored channel interaction
        delta = torch.matmul(fused, attn)
        return auxiliary + self.proj(delta)


class _TransformerFuse(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class EICILayer(nn.Module):
    """
    One EICI block: gather event + illumination into image stream, fuse, inject back.

    Implements Eq. 7–9 at feature-map resolution.
    """

    def __init__(self, channels: int, *, num_heads: int = 4) -> None:
        super().__init__()
        self.gather_e = _CovarianceGather(channels, num_heads)
        self.gather_l = _CovarianceGather(channels, num_heads)
        self.fuse = _TransformerFuse(channels, num_heads)
        self.inject_e = _BackwardInject(channels)
        self.inject_l = _BackwardInject(channels)

    def _to_tokens(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        return x.flatten(2).transpose(1, 2)

    def _to_map(self, tokens: Tensor, h: int, w: int) -> Tensor:
        b, _, c = tokens.shape
        return tokens.transpose(1, 2).reshape(b, c, h, w)

    def forward(
        self,
        image_feat: Tensor,
        event_feat: Tensor,
        illum_feat: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        b, c, h, w = image_feat.shape
        fi = self._to_tokens(image_feat)
        fe = self._to_tokens(event_feat)
        fl = self._to_tokens(illum_feat)

        fi_e, ae = self.gather_e(fe, fi)
        fi_l, al = self.gather_l(fl, fi)
        fused = self.fuse(fi + fi_e + fi_l)
        fe_out = self.inject_e(fused, ae, fe)
        fl_out = self.inject_l(fused, al, fl)

        return (
            self._to_map(fused, h, w),
            self._to_map(fe_out, h, w),
            self._to_map(fl_out, h, w),
        )
