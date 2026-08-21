"""JAFAR-style image-guided feature upsampler (Sec. 3.3)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class JafarUpsampler(nn.Module):
    """Lift low-res backbone features to ``H/2 × W/2`` with image-guided cross-attention."""

    def __init__(
        self,
        backbone_dim: int,
        internal_dim: int = 128,
        *,
        window_size: int = 0,
    ) -> None:
        super().__init__()
        self.internal_dim = internal_dim
        self.window_size = window_size
        self.img_enc = nn.Sequential(
            nn.Conv2d(3, internal_dim, 3, padding=1, bias=False),
            nn.GroupNorm(8, internal_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(internal_dim, internal_dim, 3, padding=1, bias=False),
            nn.GroupNorm(8, internal_dim),
            nn.ReLU(inplace=True),
        )
        self.enc_q = nn.Conv2d(internal_dim, internal_dim, 1)
        self.enc_k = nn.Conv2d(internal_dim, internal_dim, 1)
        self.mod_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(backbone_dim, internal_dim * 2, 1),
        )
        self.out_proj = nn.Conv2d(backbone_dim, backbone_dim, 1)

    def forward(self, source_image: Tensor, backbone_map: Tensor, *, out_h: int, out_w: int) -> Tensor:
        """Upsample backbone features to ``(out_h, out_w)``."""
        b, d, hb, wb = backbone_map.shape
        ie = self.img_enc(source_image)
        q = self.enc_q(ie)
        q = F.adaptive_avg_pool2d(q, (out_h, out_w))
        k_tilde = self.enc_k(ie)
        k_tilde = F.adaptive_avg_pool2d(k_tilde, (hb, wb))
        gamma, beta = self.mod_proj(backbone_map).chunk(2, dim=1)
        k = gamma * k_tilde + beta
        v = backbone_map
        if self.window_size > 0:
            out = _window_cross_attention(q, k, v, window=self.window_size)
        else:
            out = _global_cross_attention(q, k, v)
        return self.out_proj(out)


def _global_cross_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """``q``: [B,d,h,w], ``k/v``: [B,d,hb,wb] → [B,D,h,w]."""
    b, d, h, w = q.shape
    _, dv, hb, wb = v.shape
    qf = q.flatten(2).transpose(1, 2)  # B, hw, d
    kf = k.flatten(2).transpose(1, 2)
    vf = v.flatten(2).transpose(1, 2)
    scale = 1.0 / math.sqrt(d)
    attn = torch.softmax(qf @ kf.transpose(-1, -2) * scale, dim=-1)
    out = attn @ vf
    return out.transpose(1, 2).reshape(b, dv, h, w)


def _window_cross_attention(q: Tensor, k: Tensor, v: Tensor, *, window: int) -> Tensor:
    """Non-overlapping window attention (≈50% VRAM vs global)."""
    b, d, h, w = q.shape
    _, dv, hb, wb = v.shape
    ws = window
    pads_h = (ws - h % ws) % ws
    pads_w = (ws - w % ws) % ws
    if pads_h or pads_w:
        q = F.pad(q, (0, pads_w, 0, pads_h))
    hp, wp = q.shape[-2], q.shape[-1]
    qh = q.reshape(b, d, hp // ws, ws, wp // ws, ws).permute(0, 2, 4, 1, 3, 5)
    qh = qh.reshape(b * (hp // ws) * (wp // ws), d, ws, ws)
    # pool keys/values per window region on backbone grid
    kh = F.interpolate(k, size=(hp // ws, wp // ws), mode="bilinear", align_corners=False)
    vh = F.interpolate(v, size=(hp // ws, wp // ws), mode="bilinear", align_corners=False)
    kh = kh.unsqueeze(2).unsqueeze(4).expand(-1, -1, ws, -1, ws, -1).reshape_as(qh)
    vh = vh.unsqueeze(2).unsqueeze(4).expand(-1, -1, ws, -1, ws, -1).reshape_as(qh)
    out = _global_cross_attention(qh, kh, vh)
    out = out.reshape(b, hp // ws, wp // ws, dv, ws, ws).permute(0, 3, 1, 4, 2, 5)
    out = out.reshape(b, dv, hp, wp)
    if pads_h or pads_w:
        out = out[:, :, :h, :w]
    return out
