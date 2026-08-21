"""Cross-fusion tone mapper f = (ftm, fmix) (Sec. 4.2, Fig. 3)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ToneMappingMLP(nn.Module):
    """ftm: HDR → global-local LDR pair."""

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.global_head = nn.Sequential(
            nn.Conv2d(3, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 3, 1),
            nn.Sigmoid(),
        )
        self.local_head = nn.Sequential(
            nn.Conv2d(3, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, hdr: Tensor) -> tuple[Tensor, Tensor]:
        iglo = self.global_head(hdr)
        iloc = self.local_head(hdr)
        return iglo, iloc


class FusionMLP(nn.Module):
    """fmix: cross-fuse global-local LDR pairs (Eq. 12–13)."""

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, iglo: Tensor, iloc: Tensor) -> Tensor:
        return self.net(torch.cat([iglo, iloc], dim=1))


class ToneMapper(nn.Module):
    """Full tone mapper producing ILDR, IIG_LDR, IGI_LDR (Eq. 14)."""

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.ftm = ToneMappingMLP(hidden)
        self.fmix = FusionMLP(hidden)

    def forward(
        self,
        ihdr_scaled: Tensor,
        ihdr_relit: Tensor,
        *,
        fmix_enabled: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        iglo, iloc = self.ftm(ihdr_scaled)
        iglo_r, iloc_r = self.ftm(ihdr_relit)
        if fmix_enabled:
            iig = self.fmix(iglo, iloc_r)
            igi = self.fmix(iglo_r, iloc)
            ildr = iig + igi
        else:
            iig = iglo
            igi = iloc
            ildr = (iglo + iloc) * 0.5
        intermediates = {
            "iglo": iglo,
            "iloc": iloc,
            "iglo_relit": iglo_r,
            "iloc_relit": iloc_r,
        }
        return ildr, iig, igi, intermediates

    def freeze_fmix(self) -> None:
        for p in self.fmix.parameters():
            p.requires_grad = False

    def unfreeze_fmix(self) -> None:
        for p in self.fmix.parameters():
            p.requires_grad = True
