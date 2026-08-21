"""BTS-CAFE evaluation smoke (arXiv:2605.29862)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.btscafe.config import BTSCafeConfig
from ltx_trainer.btscafe.pipeline import pipeline_demo


def evaluation_smoke(cfg: BTSCafeConfig | None = None) -> dict[str, Any]:
    c = cfg or BTSCafeConfig()
    demo = pipeline_demo(c, seed=42)
    return {
        "paper": c.paper_arxiv,
        "backbone": c.backbone,
        "ood_score_yunting": c.ood_score_yunting,
        "gin_active": demo["gin_active_round10"],
        "text_device_neutralized": demo["text_device_neutralized"],
        "lodo_mean_score": demo["lodo_mean_score"],
    }
