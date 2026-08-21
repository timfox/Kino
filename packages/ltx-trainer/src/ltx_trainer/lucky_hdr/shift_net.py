"""Coarse + fine shift prediction (Eq. 4–5)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.lucky_hdr.features import align_features


class ShiftStage(nn.Module):
    """Predict residual flow aligning ``alt`` onto ``base`` using φ features."""

    def __init__(self, *, channels: int = 16) -> None:
        super().__init__()
        self.coarse = nn.Sequential(
            nn.Conv2d(8, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2, 3, padding=1),
        )
        self.fine = nn.Sequential(
            nn.Conv2d(8 + 2, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2, 3, padding=1),
        )

    def forward(self, base: Tensor, alt: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Returns ``(warped_alt, shift, valid)`` at full resolution."""
        from ltx_trainer.lucky_hdr.warp import warp_image

        phi_b = align_features(base)
        phi_a = align_features(alt)
        feat = torch.cat([phi_b, phi_a], dim=0).unsqueeze(0)
        coarse = self.coarse(feat)
        coarse_up = F.interpolate(coarse, size=base.shape[-2:], mode="bilinear", align_corners=False)
        warped_c, valid_c = warp_image(alt.unsqueeze(0), coarse_up)
        fine_in = torch.cat([feat, coarse_up], dim=1)
        fine = self.fine(fine_in) * 0.25
        shift = coarse_up + fine
        warped, valid = warp_image(alt.unsqueeze(0), shift)
        return warped.squeeze(0), shift.squeeze(0), (valid * valid_c).squeeze(0)
