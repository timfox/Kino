"""Stage-2 LMA-Decoder and CCP-Refiner stubs (Sec. IV-C)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.lucky_hdr.tonemap import tone_map_mu


class AttentionModulationBlock(nn.Module):
    """AMB: offset + modulator for adapter-guided decoding (Fig. 3c stub)."""

    def __init__(self, dec_ch: int, adap_ch: int) -> None:
        super().__init__()
        self.offset = nn.Conv2d(dec_ch + adap_ch, dec_ch, 3, padding=1)
        self.modulator = nn.Sequential(
            nn.Conv2d(dec_ch + adap_ch, dec_ch, 3, padding=1),
            nn.InstanceNorm2d(dec_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(dec_ch, dec_ch, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, fd: Tensor, fa: Tensor) -> Tensor:
        cat = torch.cat([fd, fa], dim=1)
        return fd + self.modulator(cat) * self.offset(cat)


class LMADecoder(nn.Module):
    """Decode latent prior + PMF features to μ-law HDR estimate (Eq. 11–12 stub)."""

    def __init__(self, latent_ch: int = 32, out_ch: int = 3) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(latent_ch, 32, 4, stride=2, padding=1)
        self.amb = AttentionModulationBlock(32, 32)
        self.head = nn.Conv2d(32, out_ch, 3, padding=1)

    def forward(self, z0: Tensor, adapter_feats: list[Tensor]) -> Tensor:
        x = F.relu(self.up(z0))
        fa = F.interpolate(adapter_feats[0], size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = self.amb(x, fa)
        return self.head(x).sigmoid()


class CCPRefiner(nn.Module):
    """Cyclic-consistent physical refiner (Eq. 13–14 stub)."""

    def __init__(self, ch: int = 3, period: float = 256.0) -> None:
        super().__init__()
        self.period = period
        self.net = nn.Sequential(
            nn.Conv2d(ch * 4, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, ch, 3, padding=1),
        )

    def cyclic_encoding(self, linear: Tensor) -> Tensor:
        phi = torch.remainder(linear, self.period) / self.period
        return torch.cat([torch.sin(2 * math.pi * phi), torch.cos(2 * math.pi * phi)], dim=1)

    def forward(self, i_mu_hat: Tensor, modulo: Tensor, linear_hat: Tensor) -> Tensor:
        if i_mu_hat.dim() == 3:
            i_mu_hat = i_mu_hat.unsqueeze(0)
            modulo = modulo.unsqueeze(0)
            linear_hat = linear_hat.unsqueeze(0)
        cyc = self.cyclic_encoding(linear_hat)
        if modulo.shape[-2:] != i_mu_hat.shape[-2:]:
            modulo = F.interpolate(modulo, size=i_mu_hat.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([i_mu_hat, modulo, cyc], dim=1)
        return (i_mu_hat + self.net(x)).clamp(0.0, 1.0)


def inverse_mu_tone(i_mu: Tensor, mu: float = 5000.0) -> Tensor:
    """Inverse of Eq. 11."""
    denom = math.log1p(mu)
    return (torch.exp(i_mu.clamp(0.0, 1.0) * denom) - 1.0) / mu
