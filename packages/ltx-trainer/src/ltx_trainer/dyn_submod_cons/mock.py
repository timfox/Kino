from __future__ import annotations

from typing import Any

from ltx_trainer.dyn_submod_cons.config import DynSubmodConsConfig
from ltx_trainer.dyn_submod_cons.pipeline import evaluation_demo, framework_card


def evaluation_smoke(*, seed: int = 0) -> dict[str, Any]:
    demo = evaluation_demo(seed=seed)
    return {
        "paper": DynSubmodConsConfig().paper_arxiv,
        "framework": framework_card()["name"],
        "approx_ratio": demo["demo"]["approx_ratio"],
        "max_consistency": demo["demo"]["max_consistency"],
    }
