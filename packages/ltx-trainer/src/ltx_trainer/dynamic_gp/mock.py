"""CPU smoke exports."""

from __future__ import annotations

from typing import Any

from ltx_trainer.dynamic_gp.paper import paper_card
from ltx_trainer.dynamic_gp.pipeline import run_demo


def evaluation_smoke() -> dict[str, Any]:
    demo = run_demo()
    summary = demo["summary"]
    return {
        "paper": paper_card(),
        "demo": demo,
        "error_reduction_anchor": summary["error_reduction_M3_to_M101"],
        "status": "ok"
        if demo["final_l2_error"] < summary["heat_error_M3"]
        and summary["error_reduction_M3_to_M101"] >= 5.0
        else "fail",
    }
