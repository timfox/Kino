"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.visa.config import VisaConfig
from ltx_trainer.visa.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    table1_mmar_modality,
    table4_leaderboard,
)
from ltx_trainer.visa.taxonomy import category_registry, routing_strategy_counts


def evaluation_smoke(cfg: VisaConfig | None = None) -> dict[str, Any]:
    cfg = cfg or VisaConfig()
    demo = evaluation_demo(seed=0, cfg=cfg)
    fw = demo["framework"]
    assert fw["paper"] == "arXiv:2606.07264"
    assert fw["headline"]["accuracy_pct"] == 77.40
    assert fw["headline"]["rubrics_pct"] == 66.23

    visa_t1 = next(r for r in table1_mmar_modality() if "VISA" in r["model"])
    assert visa_t1["avg"] == 77.4
    assert visa_t1["speech"] == 84.0

    visa_lb = next(r for r in table4_leaderboard() if r["team"] == "VISA (ours)")
    assert visa_lb["rubrics"] == 66.23
    assert visa_lb["acc"] == 77.40

    assert len(category_registry()) == 27
    counts = routing_strategy_counts()
    assert sum(counts.values()) == 27
    assert counts["llm_reasoning_and_selection"] == 13
    assert counts["vlm_empowered_spectral_reasoning"] == 6

    b = benchmarks_bundle()
    assert len(b["category_registry"]) == 27

    return {
        "status": "ok",
        "paper": fw["paper"],
        "accuracy_pct": cfg.accuracy,
        "rubrics_pct": cfg.rubrics_score,
        "mmar_avg_pct": cfg.mmar_avg_accuracy,
        "categories": cfg.num_fine_categories,
    }
