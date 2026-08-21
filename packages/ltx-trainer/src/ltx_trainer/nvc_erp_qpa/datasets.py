"""Evaluation datasets (JVET class S1, Sec. V-A)."""

from __future__ import annotations

from typing import Any

JVET_CLASS_S1: tuple[str, ...] = (
    "SkateboardInLot",
    "ChairLift",
    "KiteFlite",
    "Harbor",
    "Trolley",
    "GasLamp",
)


def datasets_card() -> dict[str, Any]:
    return {
        "eval": {
            "name": "JVET class S1 360° test sequences",
            "ref": "JVET-U2012",
            "sequences": list(JVET_CLASS_S1),
            "format": "ERP YUV 4:2:0",
            "resolution": "8K",
            "frames": 300,
            "bit_depth": [8, 10],
        },
        "codec_baselines": [
            "DCVC-RT (pretrained, LDP, PyTorch on V100)",
            "VVenC 1.12.0 slower (VVC QPA)",
            "HM-16.15 (HEVC QPA, RA, from [12])",
        ],
        "metric": "WS-PSNR (wy, wu, wv) = (6, 1, 1)",
    }
