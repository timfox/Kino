"""CPU smoke exports."""

from __future__ import annotations

from typing import Any

from ltx_trainer.flood_physics.paper import paper_card
from ltx_trainer.flood_physics.pipeline import run_demo


def evaluation_smoke() -> dict[str, Any]:
    demo = run_demo()
    summary = demo["summary"]
    return {
        "paper": paper_card(),
        "demo": demo,
        "IoU_anchor": summary["IoU"],
        "F1_anchor": summary["F1"],
        "status": "ok"
        if summary["IoU"] >= 0.80
        and summary["F1"] >= 0.88
        and summary["depth_rmse_m"] <= 0.25
        and summary["mass_imbalance_pct"] <= 2.5
        else "fail",
    }
