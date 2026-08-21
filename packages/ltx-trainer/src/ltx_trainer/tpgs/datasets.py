"""Benchmark dataset cards (Ricoh360, OmniPhotos, OmniScenes)."""

from __future__ import annotations

from typing import Any


def datasets_card() -> dict[str, Any]:
    return {
        "Ricoh360": {
            "scenes": 12,
            "capture": "in-place rotation outdoor",
            "hub": "paper supplement",
        },
        "OmniPhotos": {
            "scenes": 10,
            "capture": "selfie-stick circular motion",
            "hub": "paper supplement",
        },
        "OmniScenes": {
            "scenes": 7,
            "capture": "indoor roaming, motion blur",
            "hub": "Piccolo / OmniScenes",
        },
        "metrics": ["PSNR", "SSIM", "LPIPS"],
        "train_splits": "follow ODGS [10]",
    }
