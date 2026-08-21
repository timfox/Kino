"""LPR encoder + DHR decoder stubs (LF-Diff Sec. 3)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.lf_diff.tonemap import tonemap_for_vae


class LPENetStub(nn.Module):
    """Encode tonemapped bracket concat → low-res latent prior representation (LPR)."""

    def __init__(self, in_ch: int = 9, latent_ch: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, latent_ch, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_ch, latent_ch, 3, stride=2, padding=1),
        )

    def forward(self, bracket_tonemap: Tensor) -> Tensor:
        return self.net(bracket_tonemap)


class DHRStub(nn.Module):
    """Decode LPR + noisy latent → scene-linear HDR estimate."""

    def __init__(self, latent_ch: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_ch * 2, 48, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 3, 3, padding=1),
            nn.Softplus(),
        )

    def forward(self, lpr: Tensor, z: Tensor) -> Tensor:
        if z.shape[-2:] != lpr.shape[-2:]:
            z = torch.nn.functional.interpolate(z, size=lpr.shape[-2:], mode="bilinear", align_corners=False)
        return self.net(torch.cat([lpr, z], dim=1))


def encode_bracket_stack(model: LPENetStub, brackets: list[Tensor]) -> Tensor:
    tm_parts = [tonemap_for_vae(b) for b in brackets]
    tm = torch.cat(tm_parts, dim=0)
    return model(tm.unsqueeze(0))
