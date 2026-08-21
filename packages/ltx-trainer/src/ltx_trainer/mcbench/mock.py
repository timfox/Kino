"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mcbench.config import McbenchConfig
from ltx_trainer.mcbench.pipeline import benchmarks_bundle, pipeline_demo, table3_model_accuracy
from ltx_trainer.mcbench.taxonomy import total_samples


def evaluation_smoke(cfg: McbenchConfig | None = None) -> dict[str, Any]:
    c = cfg or McbenchConfig()
    demo = pipeline_demo(seed=0, cfg=c)

    assert c.total_scenarios == 1196
    assert total_samples() == 1196
    assert demo["matches_config_total"] is True
    assert demo["above_random"] is True
    assert demo["best_model_avg"] == 64.5
    assert demo["setting2_oversensitivity"] is True
    assert c.gemini_perception_illegal == 0.698

    top = max(r["avg_accuracy"] for r in table3_model_accuracy(c) if r["model"] != "Random")
    assert abs(top - 64.5) < 0.1

    return {
        "status": "ok",
        "paper": c.paper_arxiv,
        "framework": c.framework,
        "total_scenarios": c.total_scenarios,
        "best_avg_accuracy": demo["best_model_avg"],
        "categories": len(benchmarks_bundle(c)["taxonomy"]),
    }
