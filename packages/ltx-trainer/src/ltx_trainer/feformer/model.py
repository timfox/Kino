"""FEFormer U-shaped hierarchical ViT (Sec. 3.1)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.feformer.block import FEFormerBlock
from ltx_trainer.feformer.config import FEFormerConfig
from ltx_trainer.feformer.fcsb import FCSB
from ltx_trainer.feformer.waff import WAFF


class _Stem(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch // 2, 3, stride=2, padding=1),
            nn.BatchNorm3d(out_ch // 2),
            nn.GELU(),
            nn.Conv3d(out_ch // 2, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class FEFormer(nn.Module):
    def __init__(self, cfg: FEFormerConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or FEFormerConfig()
        c = self.cfg.base_channels
        k = min(self.cfg.dw_kernel, 3) if self.cfg.image_size <= 16 else self.cfg.dw_kernel
        self.enc_stem = _Stem(self.cfg.in_channels, c)
        self.enc1 = nn.ModuleList([FEFormerBlock(c, mlp_ratio=self.cfg.mlp_ratio, dw_kernel=k) for _ in range(2)])
        self.down1 = nn.Conv3d(c, c * 2, 3, stride=2, padding=1)
        self.enc2 = nn.ModuleList([FEFormerBlock(c * 2, mlp_ratio=self.cfg.mlp_ratio, dw_kernel=k) for _ in range(2)])
        self.down2 = nn.Conv3d(c * 2, c * 4, 3, stride=2, padding=1)
        self.enc3 = nn.ModuleList([FEFormerBlock(c * 4, mlp_ratio=self.cfg.mlp_ratio, dw_kernel=k) for _ in range(2)])
        self.bottleneck = nn.ModuleList([FEFormerBlock(c * 4, mlp_ratio=self.cfg.mlp_ratio, dw_kernel=k) for _ in range(2)])
        self.up2 = nn.ConvTranspose3d(c * 4, c * 2, 2, stride=2)
        self.waff2 = WAFF(c * 2, dw_kernel=k)
        self.dec2 = nn.ModuleList([FEFormerBlock(c * 2, mlp_ratio=self.cfg.mlp_ratio, dw_kernel=k) for _ in range(2)])
        self.up1 = nn.ConvTranspose3d(c * 2, c, 2, stride=2)
        self.waff1 = WAFF(c, dw_kernel=k)
        self.dec1 = nn.ModuleList([FEFormerBlock(c, mlp_ratio=self.cfg.mlp_ratio, dw_kernel=k) for _ in range(2)])
        self.fcsb = FCSB(c)
        self.dec_stem = _Stem(c, c)
        self.head = nn.Conv3d(c, self.cfg.num_classes, 1)

    def forward(self, x: Tensor) -> Tensor:
        s0 = self.enc_stem(x)
        e1 = s0
        for blk in self.enc1:
            e1 = blk(e1)
        e2 = self.down1(e1)
        for blk in self.enc2:
            e2 = blk(e2)
        e3 = self.down2(e2)
        for blk in self.enc3:
            e3 = blk(e3)
        b = e3
        for blk in self.bottleneck:
            b = blk(b)
        d2 = self.up2(b)
        d2 = self.waff2(e2, d2)
        for blk in self.dec2:
            d2 = blk(d2)
        d1 = self.up1(d2)
        d1 = self.waff1(e1, d1)
        for blk in self.dec1:
            d1 = blk(d1)
        bridge1, bridge2 = self.fcsb(e1, d1)
        d1 = d1 + bridge2
        out = self.dec_stem(d1)
        if bridge1.shape[-3:] != out.shape[-3:]:
            bridge1 = F.interpolate(bridge1, size=out.shape[-3:], mode="trilinear", align_corners=False)
        out = out + bridge1
        out = F.interpolate(out, size=x.shape[-3:], mode="trilinear", align_corners=False)
        return self.head(out)
