"""Cross-scale Multi-receptive-field Perception — CMP (Sec. III-C)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class _DeformableConvStub(nn.Module):
    """DCN stub: depthwise 3×3 with learnable offset scale."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.offset = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.dw(x) + 0.1 * self.offset(x)


class CMPBlock(nn.Module):
    """Per-stage CMP: DCN + local attention + global multiscale branch."""

    def __init__(self, in_ch: int, out_ch: int = 512) -> None:
        super().__init__()
        self.dcn = _DeformableConvStub(in_ch)
        self.local_proj = nn.Conv2d(in_ch, in_ch * 3, 1)
        self.local_dw = nn.Conv2d(in_ch * 3, in_ch * 3, 3, padding=2, dilation=2, groups=in_ch * 3)
        self.local_out = nn.Conv2d(in_ch, out_ch, 1)
        self.global_norm = nn.LayerNorm([in_ch])
        self.global_reduce = nn.Linear(in_ch, max(in_ch // 4, 16))
        self.dw5 = nn.Conv2d(max(in_ch // 4, 16), max(in_ch // 4, 16), 5, padding=2, groups=max(in_ch // 4, 16))
        self.dw7 = nn.Conv2d(max(in_ch // 4, 16), max(in_ch // 4, 16), 7, padding=3, groups=max(in_ch // 4, 16))
        self.dw9 = nn.Conv2d(max(in_ch // 4, 16), max(in_ch // 4, 16), 9, padding=4, groups=max(in_ch // 4, 16))
        self.global_out = nn.Conv2d(max(in_ch // 4, 16), out_ch, 1)
        self.in_ch = in_ch

    def forward(self, feat: Tensor) -> Tensor:
        d = F.gelu(self.dcn(feat))
        b, c, h, w = d.shape
        # local branch (simplified channel attention)
        loc = self.local_proj(d)
        loc = self.local_dw(loc)
        q, k, v = loc.chunk(3, dim=1)
        q = F.normalize(q.flatten(2), dim=1)
        k = F.normalize(k.flatten(2), dim=1)
        v = v.flatten(2)
        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) / (c**0.5), dim=-1)
        loc_out = torch.bmm(attn, v.transpose(1, 2)).transpose(1, 2).view(b, c, h, w)
        f_l = self.local_out(loc_out)
        # global branch
        g = d.permute(0, 2, 3, 1)
        g = self.global_norm(g).permute(0, 3, 1, 2)
        r = self.global_reduce(g.mean(dim=(2, 3))).unsqueeze(-1).unsqueeze(-1)
        r = r.expand(-1, -1, h, w)
        y = (self.dw5(r) + self.dw7(r) + self.dw9(r)) / 3.0
        f_g = self.global_out(y)
        if f_g.shape[-2:] != f_l.shape[-2:]:
            f_g = F.interpolate(f_g, size=f_l.shape[-2:], mode="bilinear", align_corners=False)
        return f_l + f_g


class CMPStack(nn.Module):
    def __init__(self, stage_dims: tuple[int, ...], cmp_dim: int = 512) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(CMPBlock(d, cmp_dim) for d in stage_dims)

    def forward(self, feats: list[Tensor]) -> list[Tensor]:
        return [block(f) for block, f in zip(self.blocks, feats, strict=True)]
