"""DCVC-RT + latitude QPA stub (no pretrained DCVC weights)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.nvc_erp_qpa.config import NvcErpQpaConfig
from ltx_trainer.nvc_erp_qpa.quality_parameter import latitude_q_map
from ltx_trainer.nvc_erp_qpa.vector_bank import VectorBankModulator


class DcvcRtQpaStub(nn.Module):
    """Encode path: latent → per-latitude vector modulation (Ve, Vd, Vr, Vf)."""

    def __init__(self, cfg: NvcErpQpaConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or NvcErpQpaConfig()
        c = self.cfg.latent_channels
        self.latent_enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, c, 3, padding=1),
        )
        self.latent_dec = nn.Sequential(
            nn.Conv2d(c, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
        )
        self.mod_e = VectorBankModulator(self.cfg, bank_role="encoder")
        self.mod_d = VectorBankModulator(self.cfg, bank_role="decoder")
        self.q_map: Tensor | None = None

    def encode_latent(self, frame: Tensor) -> Tensor:
        return self.latent_enc(frame)

    def forward(self, frame: Tensor) -> dict[str, Tensor]:
        """Single-frame stub: baseline q0 vs latitude-adaptive q̃φ."""
        z = self.encode_latent(frame)
        h = z.shape[-2]
        q_map = latitude_q_map(h, self.cfg)
        self.q_map = q_map

        z_base = self.mod_e(z, torch.full_like(q_map, self.cfg.q0))
        z_adapt = self.mod_e(z, q_map)
        z_dec = self.mod_d(z_adapt, q_map)
        recon = self.latent_dec(z_dec).clamp(0.0, 1.0)
        recon_base = self.latent_dec(self.mod_d(z_base, torch.full_like(q_map, self.cfg.q0))).clamp(
            0.0, 1.0
        )
        return {
            "latent": z,
            "q_map": q_map,
            "recon_qpa": recon,
            "recon_baseline": recon_base,
        }
