from __future__ import annotations

from typing import Any

from ltx_trainer.omcm_general.config import OmcmGeneralConfig
from ltx_trainer.omcm_general.pipeline import evaluation_demo, framework_card


def evaluation_smoke(*, seed: int = 0) -> dict[str, Any]:
    demo = evaluation_demo(seed=seed)
    return {
        "paper": OmcmGeneralConfig().paper_arxiv,
        "framework": framework_card()["name"],
        "matched_pairs": demo["simulation"]["matched_pairs"],
        "total_cost": round(demo["simulation"]["total_cost"], 4),
    }
