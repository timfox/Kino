"""Training / eval datasets card."""

from __future__ import annotations

from typing import Any

from ltx_trainer.cubediff.benchmarks import TRAIN_PANORAMAS


def datasets_card() -> dict[str, Any]:
    return {
        "train_sources": ["Polyhaven", "Humus", "Structured3D", "Pano360"],
        "train_panoramas": TRAIN_PANORAMAS,
        "eval": ["Laval Indoor", "SUN360"],
        "captioning": "Gemini single or per-face captions",
        "resolution": "512×1024",
    }
