"""Synthetic SDR degradations for ITM training stub (Eq. 19)."""

from __future__ import annotations

import random

import torch
from torch import Tensor

from ltx_trainer.lumaflux.color import pq_oetf


def _reinhard_tmo(hdr: Tensor, key: float = 0.18) -> Tensor:
    return hdr / (1.0 + hdr / key)


def _youtube_logc(hdr: Tensor) -> Tensor:
    return torch.log1p(hdr * 5.0) / torch.log1p(torch.tensor(5.0))


TMO_FUNCS = {
    "Reinhard": _reinhard_tmo,
    "YouTube-LogC": _youtube_logc,
    "BT2446c+GM": lambda x: _reinhard_tmo(x, 0.22),
    "OCIOv2": lambda x: x**0.45,
}


def quantize_8bit(x: Tensor, crf: int = 23) -> Tensor:
    """Simulate codec quantization; higher CRF → coarser steps."""
    levels = max(32, 256 - crf * 4)
    return (x * levels).round() / levels


def degrade_hdr_to_sdr(hdr: Tensor, *, tmo: str = "Reinhard", crf: int = 23) -> Tensor:
    """xsdr = Q_codec ∘ M2020→709 ∘ TMO(xpq) simplified stub."""
    fn = TMO_FUNCS.get(tmo, _reinhard_tmo)
    mapped = fn(hdr.clamp(min=0.0))
    sdr = mapped.clamp(0.0, 1.0)
    sdr = quantize_8bit(sdr, crf)
    return sdr


def synthesize_pair(
    size: int = 64,
    *,
    seed: int = 0,
    tmo: str | None = None,
    crf: int = 31,
) -> tuple[Tensor, Tensor]:
    g = torch.Generator().manual_seed(seed)
    hdr = torch.rand(3, size, size, generator=g) * 0.8 + 0.1
    # Highlight patch
    hdr[0, size // 4 : size // 4 + 8, size // 2 : size // 2 + 8] = 2.5
    hdr = pq_oetf(hdr.clamp(0.0, 1.0))
    tmo_name = tmo or random.choice(list(TMO_FUNCS.keys()))
    sdr = degrade_hdr_to_sdr(hdr, tmo=tmo_name, crf=crf)
    return sdr.unsqueeze(0), hdr.unsqueeze(0)
