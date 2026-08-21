"""Utility-Aware CLIP evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.uaclip.config import UAClipConfig
from ltx_trainer.uaclip.pipeline import evaluation_demo


def evaluation_smoke(cfg: UAClipConfig | None = None) -> dict[str, Any]:
    _ = cfg
    demo = evaluation_demo(seed=0, platform="amazon")
    return {
        "paper": "arXiv:2605.28733",
        "clip_infonce_loss": demo["clip_infonce_loss"],
        "utility_aware_infonce_loss": demo["utility_aware_infonce_loss"],
        "paper_uag_demand_amazon": demo["paper_uag_demand"],
        "human_selection_amazon": demo["human_selection_amazon"],
    }
