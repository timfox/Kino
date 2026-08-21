"""Wavelet-guided Adaptive Feature Fusion — WAFF (Sec. 3.5, Eq. 25–30)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.feformer.wavelet import SUBBAND_NAMES, dwt3, idwt3


class WAFF(nn.Module):
    def __init__(self, channels: int, *, dw_kernel: int = 7) -> None:
        super().__init__()
        pad = dw_kernel // 2
        self.fuse = nn.Conv3d(channels * 2, channels, 1)
        self.dw = nn.Conv3d(2, 2, dw_kernel, padding=pad, groups=2)

    def _fuse_band(self, a: Tensor, b: Tensor) -> Tensor:
        cat = torch.cat([a, b], dim=1)
        avp = cat.mean(dim=1, keepdim=True)
        mp = cat.amax(dim=1, keepdim=True)
        w = torch.sigmoid(self.dw(torch.cat([avp, mp], dim=1)))
        w1, w2 = w[:, :1], w[:, 1:2]
        if w1.shape[-3:] != a.shape[-3:]:
            w1 = F.interpolate(w1, size=a.shape[-3:], mode="trilinear", align_corners=False)
            w2 = F.interpolate(w2, size=a.shape[-3:], mode="trilinear", align_corners=False)
        return w1 * a + w2 * b

    def forward(self, x_enc: Tensor, x_dec: Tensor) -> Tensor:
        if x_enc.shape[-3:] != x_dec.shape[-3:]:
            x_dec = F.interpolate(x_dec, size=x_enc.shape[-3:], mode="trilinear", align_corners=False)
        b1, b2 = dwt3(x_enc), dwt3(x_dec)
        fused = {name: self._fuse_band(b1[name], b2[name]) for name in SUBBAND_NAMES if name in b1 and name in b2}
        out = idwt3(fused, target_shape=x_dec.shape[-3:])
        if out.shape[1] != x_dec.shape[1]:
            out = self.fuse(torch.cat([out, x_enc], dim=1))
        return out + x_dec
