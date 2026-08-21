"""evaluation_smoke for validate_paper_stubs."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mosm_dp.config import MosmDpConfig
from ltx_trainer.mosm_dp.pipeline import evaluation_demo, framework_card


def evaluation_smoke(*, seed: int = 0) -> dict[str, Any]:
    demo = evaluation_demo(seed=seed, k=8)
    card = framework_card()
    return {
        "paper": MosmDpConfig().paper_arxiv,
        "framework": card["name"],
        "F_minimax": demo["dp_multi_greedy"]["F"],
        "picked_count": len(demo["dp_multi_greedy"]["picked"]),
        "algorithms": card["algorithms"],
    }
