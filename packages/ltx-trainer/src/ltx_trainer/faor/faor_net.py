"""FAOR full network."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.faor.config import FaorConfig
from ltx_trainer.faor.geodesic import geodesic_latent_interp
from ltx_trainer.faor.safe import SAFEEncoder
from ltx_trainer.faor.sgif import SGIF
from ltx_trainer.faor.stretching import stretching_ratio_map


class FaorStub(nn.Module):
    def __init__(self, cfg: FaorConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or FaorConfig()
        self.safe = SAFEEncoder(self.cfg)
        self.sgif = SGIF(self.cfg.latent_dim)

    def forward(self, lr: Tensor, *, scale: float = 2.0) -> dict[str, Tensor]:
        h, w = lr.shape[-2:]
        md = stretching_ratio_map(h, w, device=lr.device).unsqueeze(0).unsqueeze(0)
        z = self.safe(lr, md=md, ms=None)
        if self.cfg.use_geodesic:
            z_hr = geodesic_latent_interp(z, scale)
        else:
            z_hr = torch.nn.functional.interpolate(
                z, scale_factor=scale, mode="bilinear", align_corners=False
            )
        hh, ww = z_hr.shape[-2:]
        ys = torch.linspace(-1.5708, 1.5708, hh, device=lr.device)
        xs = torch.linspace(-3.14159, 3.14159, ww, device=lr.device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([gy.reshape(-1), gx.reshape(-1)], dim=-1).unsqueeze(0)
        y = self.sgif(z_hr, coords)
        return {"sr": y, "latent": z_hr}
