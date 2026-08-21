"""VeriTrip evaluation smoke (arXiv:2605.28683)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.veritrip.config import VeriTripConfig
from ltx_trainer.veritrip.pipeline import evaluation_demo


def evaluation_smoke(cfg: VeriTripConfig | None = None) -> dict[str, Any]:
    _ = cfg
    demo = evaluation_demo()
    return {
        "paper": "arXiv:2605.28683",
        "gold_FR": demo["gold_plan"]["FR"],
        "hallucinated_FR": demo["hallucinated_plan"]["FR"],
        "mrb_documents": demo["mrb"]["documents"],
    }
