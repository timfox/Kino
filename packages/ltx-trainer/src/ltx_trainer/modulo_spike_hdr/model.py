"""Iteration-free modulo unwrapping model (Fig. 3)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.modulo_spike_hdr.pmf_adapter import PMFAdapter
from ltx_trainer.modulo_spike_hdr.unwrap_net import CCPRefiner, LMADecoder, inverse_mu_tone


@dataclass
class ModuloSpikeHdrConfig:
    period: float = 256.0
    mu: float = 5000.0
    latent_ch: int = 32


class ModuloSpikeHdrUnwrapper(nn.Module):
    """Two-stage unwrap: PMF-Adapter → latent prior → LMA-Decoder → CCP-Refiner."""

    def __init__(self, cfg: ModuloSpikeHdrConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModuloSpikeHdrConfig()
        self.adapter = PMFAdapter(period=self.cfg.period)
        self.latent_proj = nn.Sequential(
            nn.Conv2d(32, self.cfg.latent_ch, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = LMADecoder(latent_ch=self.cfg.latent_ch)
        self.refiner = CCPRefiner(period=self.cfg.period)

    def forward(self, modulo: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """modulo [3,H,W] in [0,1]. Returns (i_mu, i_linear, i_mu_coarse)."""
        squeeze = modulo.dim() == 3
        if squeeze:
            modulo = modulo.unsqueeze(0)
        feats = self.adapter(modulo)
        z0 = self.latent_proj(feats[0])
        i_mu_coarse = self.decoder(z0, feats)
        linear_coarse = inverse_mu_tone(i_mu_coarse, self.cfg.mu)
        i_mu = self.refiner(i_mu_coarse, modulo, linear_coarse)
        i_lin = inverse_mu_tone(i_mu, self.cfg.mu)
        if squeeze:
            i_mu = i_mu.squeeze(0)
            i_lin = i_lin.squeeze(0)
            i_mu_coarse = i_mu_coarse.squeeze(0)
        return i_mu, i_lin, i_mu_coarse
