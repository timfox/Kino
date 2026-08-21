"""Two-level Channel-wise Wavelet Packet ChWP (Eq. 10)."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ltx_trainer.chwdta.config import ChwdtaConfig
from ltx_trainer.chwdta.lifting import LearnableLifting1D


def chwp_decompose(y: Tensor, lift: LearnableLifting1D) -> dict[str, Tensor]:
    """(L1,H1)=WTc(Y); (LL,LH)=WTc(L1); (HL,HH)=WTc(H1)."""
    l1, h1 = lift(y)
    ll, lh = lift(l1)
    hl, hh = lift(h1)
    return {"LL": ll, "LH": lh, "HL": hl, "HH": hh}


def slice_layout(
    subbands: dict[str, Tensor],
    *,
    slices_per_subband: int = 2,
) -> list[Tensor]:
    """Split each equal-sized subband into channel slices for ChARM."""
    slices: list[Tensor] = []
    order = ("LL", "LH", "HL", "HH")
    for name in order:
        band = subbands[name]
        c = band.shape[1]
        if slices_per_subband <= 1:
            slices.append(band)
        else:
            step = c // slices_per_subband
            for i in range(slices_per_subband):
                lo = i * step
                hi = (i + 1) * step if i < slices_per_subband - 1 else c
                slices.append(band[:, lo:hi])
    return slices


def chwp_summary(cfg: ChwdtaConfig | None = None) -> dict[str, Any]:
    cfg = cfg or ChwdtaConfig()
    lift = LearnableLifting1D(cfg.latent_channels)
    y = torch.randn(1, cfg.latent_channels, 16, 16)
    sub = chwp_decompose(y, lift)
    slices_8 = slice_layout(sub, slices_per_subband=2)
    slices_4 = slice_layout(sub, slices_per_subband=1)
    return {
        "subbands": list(cfg.chwp_subbands),
        "subband_shapes": {k: list(v.shape) for k, v in sub.items()},
        "num_slices_8": len(slices_8),
        "num_slices_4": len(slices_4),
    }
