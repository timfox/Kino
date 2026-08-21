from __future__ import annotations

from typing import Any

from ltx_trainer.dyn_quasi_clique.config import DynQuasiCliqueConfig
from ltx_trainer.dyn_quasi_clique.pipeline import evaluation_demo, framework_card


def evaluation_smoke(*, seed: int = 0) -> dict[str, Any]:
    demo = evaluation_demo(seed=seed)
    return {
        "paper": DynQuasiCliqueConfig().paper_arxiv,
        "framework": framework_card()["name"],
        "static_clique_size": demo["static_clique_size"],
        "dynamic_final": demo["dynamic_trace"][-1] if demo["dynamic_trace"] else 0,
    }
