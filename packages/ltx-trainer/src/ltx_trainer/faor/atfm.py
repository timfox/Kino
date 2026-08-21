"""Affine-Transformation-based Feature Modulating (Eq. 6)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.faor.stretching import stretching_ratio_map


class ATFM(nn.Module):
    """fat = alpha ⊙ fln + beta from Md and optional Ms."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.prior_enc = nn.Sequential(
            nn.Conv2d(2, ch // 4, 1),
            nn.SiLU(),
            nn.Conv2d(ch // 4, ch * 2, 1),
        )

    def forward(
        self,
        fln: Tensor,
        *,
        md: Tensor | None = None,
        ms: Tensor | None = None,
    ) -> Tensor:
        b, c, h, w = fln.shape
        if md is None:
            md = stretching_ratio_map(h, w, device=fln.device).unsqueeze(0).unsqueeze(0)
        if md.shape[-2:] != (h, w):
            md = torch.nn.functional.interpolate(md, size=(h, w), mode="bilinear", align_corners=False)
        if ms is None:
            ms = torch.zeros_like(md)
        if ms.shape[-2:] != (h, w):
            ms = torch.nn.functional.interpolate(ms, size=(h, w), mode="nearest")
        priors = torch.cat([md / 255.0, ms], dim=1)
        ab = self.prior_enc(priors)
        alpha, beta = ab.chunk(2, dim=1)
        return alpha * fln + beta
