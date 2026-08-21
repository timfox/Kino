"""Open challenges taxonomy (Sec. VIII, Fig. 14, Table VI)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.predictive_autoscaling.benchmarks import table_vi_open_challenges


def future_research_card() -> dict[str, Any]:
    return {
        "themes": [
            "forecasting_and_prediction",
            "fl_system_challenges",
            "large_model_multimodal",
            "kubernetes_orchestration",
            "resource_cost_privacy",
        ],
        "directions": table_vi_open_challenges(),
        "unified_goal": (
            "self-correcting, uncertainty-aware, cross-layer, autonomously optimising autoscaling"
        ),
    }
