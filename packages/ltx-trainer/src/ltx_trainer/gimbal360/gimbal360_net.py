"""Gimbal360 composite module (Fig. 2)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.gimbal360.auto_leveling import DifferentiableAutoLeveling
from ltx_trainer.gimbal360.config import Gimbal360Config
from ltx_trainer.gimbal360.fill_conditioning import concat_fill_channels, fill_channel_count
from ltx_trainer.gimbal360.topology import roll_azimuth, siamese_shift_loss


class Gimbal360CompletionStub(nn.Module):
    """Auto-leveling + latent denoising stub for smoke training."""

    def __init__(self, cfg: Gimbal360Config | None = None) -> None:
        super().__init__()
        self.cfg = cfg or Gimbal360Config()
        c = self.cfg.latent_channels
        self.dal = DifferentiableAutoLeveling()
        self.denoise = nn.Sequential(
            nn.Conv2d(fill_channel_count(c), 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, c, 3, padding=1),
        )

    def _predict_eps(self, z_t: Tensor, mask: Tensor, z_canonical: Tensor) -> Tensor:
        return self.denoise(concat_fill_channels(z_t, mask, z_canonical))

    def forward(
        self,
        perspective: Tensor,
        mask: Tensor,
        z_ref: Tensor,
        *,
        shift_delta: int = 8,
    ) -> dict[str, Tensor]:
        flow, _rot, z_canonical = self.dal(perspective)
        if z_canonical.shape[-2:] != mask.shape[-2:]:
            z_canonical = nn.functional.interpolate(
                z_canonical, size=mask.shape[-2:], mode="bilinear", align_corners=False
            )
        z_t = z_ref
        eps_base = self._predict_eps(z_t, mask, z_canonical)
        eps_shifted = self._predict_eps(
            roll_azimuth(z_t, shift_delta),
            roll_azimuth(mask, shift_delta),
            roll_azimuth(z_canonical, shift_delta),
        )
        return {
            "flow": flow,
            "z_canonical": z_canonical,
            "eps_base": eps_base,
            "eps_shifted": eps_shifted,
        }
