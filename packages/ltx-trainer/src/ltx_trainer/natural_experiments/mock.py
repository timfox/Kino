"""CPU smoke for paper stub validation."""

from __future__ import annotations

from typing import Any

from ltx_trainer.natural_experiments.pipeline import evaluation_demo, framework_card


def evaluation_smoke() -> dict[str, Any]:
    demo = evaluation_demo()
    assert demo["framework"]["natural_experiment_count"] == 3
    assert len(demo["table2_datasets"]) == 11
    ne = [r for r in demo["table3_f1"] if r.get("natural_exp")]
    assert len(ne) >= 3
    assert demo["downstream_smoke"]["sachs_test_f1"]["ISK"]["I"] > demo["downstream_smoke"]["sachs_test_f1"]["O"]["I"]
    dcdi = demo["dcdi_smoke"]["modes"]["ISK"]
    assert dcdi["mb_edit_distance"] <= 2
    return {
        "status": "ok",
        "paper": demo["framework"]["paper"],
        "natural_experiment_datasets": demo["framework"]["natural_experiment_datasets"],
        "causal_isk_mb": dcdi["mb_indices"],
    }


__all__ = ["evaluation_smoke", "framework_card"]
