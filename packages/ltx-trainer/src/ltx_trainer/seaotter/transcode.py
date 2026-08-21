"""One-time cloud transcode: G_S output → standard JPEG artifact (§2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ltx_trainer.seaotter.color_transform import LearnedColorTransform
from ltx_trainer.seaotter.config import SeaotterConfig
from ltx_trainer.seaotter.frappe import FrappeDecoder, FrappeEncoder
from ltx_trainer.seaotter.jpeg_sandwich import JpegSandwich
from ltx_trainer.seaotter.quantization import LearnedQuantizationBank


@dataclass
class TranscodeResult:
    transmit_bpp: float
    storage_bpp: float
    transmit_cr: float
    storage_cr: float
    variant: str  # ZS | FT


def compression_ratio(bpp: float) -> float:
    """CR = 24 / bpp for RGB 8-bit."""
    return 24.0 / max(bpp, 1e-9)


def seaotter_forward(
    x: Tensor,
    *,
    n_channels: int = 12,
    variant: str = "ZS",
    cfg: SeaotterConfig | None = None,
) -> tuple[Tensor, TranscodeResult]:
    """
    Full pipeline smoke: encode latent → decode RGB → JPEG sandwich → RGB out.

    Storage bpp uses paper Table 1 anchors for ZS/FT at n=12 (cls).
    """
    cfg = cfg or SeaotterConfig()
    enc = FrappeEncoder(cfg)
    dec = FrappeDecoder(cfg)
    sandwich = JpegSandwich(cfg)

    z = enc(x, n_channels=n_channels)
    rgb_mid = dec(z.float())
    recon, bpp_proxy = sandwich.forward_rate(rgb_mid, rate_idx=1)

    # Paper Table 1 cls transmit/storage @ n=12
    tbpp = cfg.matched_transmit_bpp_cls
    storage = {
        "ZS": 1.2822,
        "FT": 0.9046,
        "FRAPPE": 0.1086,
    }.get(variant, 1.2822)

    meta = TranscodeResult(
        transmit_bpp=tbpp,
        storage_bpp=storage,
        transmit_cr=compression_ratio(tbpp),
        storage_cr=compression_ratio(storage),
        variant=variant,
    )
    return recon, meta


def transcode_smoke(cfg: SeaotterConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SeaotterConfig()
    x = torch.randn(1, 3, 96, 96)
    _, meta_zs = seaotter_forward(x, variant="ZS", cfg=cfg)
    _, meta_ft = seaotter_forward(x, variant="FT", cfg=cfg)
    return {
        "zs_storage_cr": meta_zs.storage_cr,
        "ft_storage_cr": meta_ft.storage_cr,
        "transmit_cr": meta_zs.transmit_cr,
    }
