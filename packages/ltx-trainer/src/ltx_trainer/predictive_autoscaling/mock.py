"""CPU smoke exports for validate_paper_stubs."""

from __future__ import annotations

from typing import Any

from ltx_trainer.predictive_autoscaling.drift import autoscaling_drift_index
from ltx_trainer.predictive_autoscaling.paper import knowledge_bundle, paper_card
from ltx_trainer.predictive_autoscaling.pipeline import run_demo


def evaluation_smoke() -> dict[str, Any]:
    demo = run_demo()
    adi = autoscaling_drift_index([0.8, 0.85, 0.9], [0.75, 0.82, 0.88])
    return {
        "paper": paper_card(),
        "demo": demo,
        "adi_smoke": adi,
        "tables": {
            "table_i_rows": len(knowledge_bundle()["table_i"]),
            "table_iii_rows": len(knowledge_bundle()["table_iii"]),
        },
        "status": "ok",
    }
