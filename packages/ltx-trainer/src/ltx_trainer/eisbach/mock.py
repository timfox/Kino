"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.eisbach.config import EisbachConfig
from ltx_trainer.eisbach.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    table2_structural_dimensions,
    testable_predictions,
)


def evaluation_smoke(cfg: EisbachConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EisbachConfig()
    demo = evaluation_demo(seed=0, cfg=cfg)
    fw = demo["framework"]
    assert fw["framework"] == "Eisbach"
    assert fw["barrier_lambda"] == 0.5
    assert demo["barrier"]["structured_gets_higher_weight"]

    dyn = next(r for r in table2_structural_dimensions() if "Dynamic range" in r["dimension"])
    assert ">40" in dyn["barrier"]
    assert "<25" in dyn["baseline"]

    assert len(testable_predictions()) == 5
    b = benchmarks_bundle()
    assert len(b["table1_curation"]) == 5

    return {
        "status": "ok",
        "paper": fw["paper"],
        "barrier_lambda": cfg.barrier_lambda,
        "dynamic_range_gain_db": cfg.barrier_dynamic_range_db - cfg.baseline_dynamic_range_db,
    }
