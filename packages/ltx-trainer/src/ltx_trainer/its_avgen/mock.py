from __future__ import annotations

from typing import Any

from ltx_trainer.its_avgen.config import ItsAvgenConfig
from ltx_trainer.its_avgen.pipeline import evaluation_demo, framework_card


def evaluation_smoke(*, seed: int = 0) -> dict[str, Any]:
    demo = evaluation_demo(seed=seed, n=4)
    return {
        "paper": ItsAvgenConfig().paper_arxiv,
        "framework": framework_card()["name"],
        "winner_combined": demo["best_of_n"]["winner"]["scores"]["combined"],
        "evo_best": demo["evo_search"]["best_score"],
    }
