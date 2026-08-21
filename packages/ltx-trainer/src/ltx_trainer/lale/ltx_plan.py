"""LTX / GOPEX hooks for aerial and ERP land-cover segmentation."""

from __future__ import annotations

from typing import Any

from ltx_trainer.lale.config import LaleConfig


def ltx_integration_plan(cfg: LaleConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LaleConfig()
    return {
        "paper": cfg.paper_arxiv,
        "use_cases": [
            "Sphere360 / aerial frame land-cover masks at 256²",
            "Lightweight semantic prior for LTX conditioning (low GMAC decoder path)",
            "Throughput-friendly edge deployment vs UPerNet-style heads",
        ],
        "hooks": {
            "preprocess": "256×256 RGB normalize; optional ARAS400k class map",
            "train": f"Dice loss, {cfg.train_images} images, H100 ~{cfg.train_hours_per_model_h100}h/model",
            "infer": "LALE-S1 for speed; S2-K3-PT for best efficiency-accuracy",
        },
        "variants": list(cfg.scales),
    }
