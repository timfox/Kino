"""Consistency U-Net f_θ (Sec. 3.4) with time embedding."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.expo_cm.trajectory import EACTConfig, sample_eact_state


def sinusoidal_time_embed(t: Tensor, dim: int = 64) -> Tensor:
    if t.ndim == 0:
        t = t.reshape(1)
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
    args = t.reshape(-1, 1) * freqs.reshape(1, -1)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class _TimeResBlock(nn.Module):
    def __init__(self, ch: int, tdim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, ch), ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, ch), ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.tproj = nn.Linear(tdim, ch * 2)

    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.tproj(temb).chunk(2, dim=-1)
        h = self.norm2(h) * (1.0 + scale.view(-1, h.shape[1], 1, 1)) + shift.view(-1, h.shape[1], 1, 1)
        h = self.conv2(F.silu(h))
        return x + h


class ConsistencyUNet(nn.Module):
    """U-Net predicting clean HDR ``x0`` from noisy state ``x_t`` and LDR condition ``y0``."""

    def __init__(self, *, base_ch: int = 32, tdim: int = 64) -> None:
        super().__init__()
        self.tdim = tdim
        self.t_mlp = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4
        self.in_conv = nn.Conv2d(6, c1, 3, padding=1)
        self.down1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.rb1 = _TimeResBlock(c2, tdim)
        self.down2 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)
        self.mid = _TimeResBlock(c3, tdim)
        self.up2 = nn.ConvTranspose2d(c3, c2, 4, stride=2, padding=1)
        self.rb2 = _TimeResBlock(c2, tdim)
        self.up1 = nn.ConvTranspose2d(c2, c1, 4, stride=2, padding=1)
        self.rb3 = _TimeResBlock(c1, tdim)
        self.out = nn.Conv2d(c1, 3, 3, padding=1)

    def forward(self, xt: Tensor, t: Tensor, y0: Tensor) -> Tensor:
        if xt.dim() == 3:
            xt = xt.unsqueeze(0)
            y0 = y0.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        b = xt.shape[0]
        if t.ndim == 0:
            t = t.reshape(1).expand(b)
        temb = self.t_mlp(sinusoidal_time_embed(t, self.tdim))
        x = torch.cat([xt, y0], dim=1)
        h0 = F.silu(self.in_conv(x))
        h1 = F.silu(self.down1(h0))
        h1 = self.rb1(h1, temb)
        h2 = F.silu(self.down2(h1))
        h2 = self.mid(h2, temb)
        h = F.silu(self.up2(h2) + h1)
        h = self.rb2(h, temb)
        h = F.silu(self.up1(h) + h0)
        h = self.rb3(h, temb)
        out = self.out(h).clamp(0.0, 1.0)
        return out.squeeze(0) if squeeze else out


@dataclass
class ExpoCMConfig:
    t_max: float = 1.0
    base_ch: int = 32
    eact: EACTConfig | None = None


class ExpoCM(nn.Module):
    """One-step exposure-aware consistency HDR model (arXiv:2605.02464)."""

    def __init__(self, cfg: ExpoCMConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ExpoCMConfig()
        self.net = ConsistencyUNet(base_ch=self.cfg.base_ch)
        self.eact = self.cfg.eact or EACTConfig(t_max=self.cfg.t_max)

    def predict(self, xt: Tensor, t: Tensor | float, y0: Tensor) -> Tensor:
        t_t = t if isinstance(t, Tensor) else torch.tensor(float(t), device=xt.device, dtype=xt.dtype)
        return self.net(xt, t_t, y0)

    def one_step(self, ldr: Tensor, *, x0_hint: Tensor | None = None) -> Tensor:
        """Single inference step at ``t=t_max`` (Eq. 4 inference)."""
        if ldr.dim() == 3:
            ldr = ldr.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        x0 = x0_hint if x0_hint is not None else ldr
        if x0.dim() == 3:
            x0 = x0.unsqueeze(0)
        xt, _ = sample_eact_state(x0, ldr, self.cfg.t_max, cfg=self.eact)
        t = torch.full((ldr.shape[0],), self.cfg.t_max, device=ldr.device, dtype=ldr.dtype)
        hdr = self.net(xt, t, ldr)
        return hdr.squeeze(0) if squeeze else hdr

    def forward(self, ldr: Tensor) -> Tensor:
        return self.one_step(ldr)
