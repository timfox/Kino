"""Pipeline for dynamic consistent submodular maximization."""

from __future__ import annotations

from typing import Any

from ltx_trainer.dyn_submod_cons.config import DynSubmodConsConfig
from ltx_trainer.dyn_submod_cons.scheduling import (
    dynamic_consistent_cardinality_smoke,
    random_scheduling_levels,
    toy_coverage_instance,
)


def framework_card(cfg: DynSubmodConsConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DynSubmodConsConfig()
    return {
        "name": "DynSubmodCons",
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "modules": ["RANDOM-SCHEDULING", "Robust rotation", "Dynamic greedy wrapper"],
        "cardinality": {
            "approx": cfg.cardinality_approx,
            "consistency": cfg.cardinality_consistency,
        },
        "matroid": {
            "approx": cfg.matroid_approx,
            "consistency": cfg.matroid_consistency,
        },
    }


def table_guarantees(cfg: DynSubmodConsConfig | None = None) -> list[dict[str, str]]:
    cfg = cfg or DynSubmodConsConfig()
    return [
        {
            "constraint": "cardinality k",
            "approx": cfg.cardinality_approx,
            "consistency": cfg.cardinality_consistency,
        },
        {
            "constraint": "matroid rank k",
            "approx": cfg.matroid_approx,
            "consistency": cfg.matroid_consistency,
        },
    ]


def benchmarks_bundle() -> dict[str, Any]:
    return {"guarantees": table_guarantees(), "prior": "insertion-only ~0.51 randomized (Dütting et al.)"}


def evaluation_demo(*, seed: int = 0, k: int = 5) -> dict[str, Any]:
    covers = toy_coverage_instance(seed=seed)
    stream = [("ins", i) for i in range(8)] + [("del", 2), ("del", 5), ("ins", 9)]
    out = dynamic_consistent_cardinality_smoke(covers, stream, k, epsilon=0.1, rng=__import__("numpy").random.default_rng(seed))
    return {
        "paper": DynSubmodConsConfig().paper_arxiv,
        "demo": out,
        "schedule_preview": random_scheduling_levels(k, 3, epsilon=0.1),
    }
