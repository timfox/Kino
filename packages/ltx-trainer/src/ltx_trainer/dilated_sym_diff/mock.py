"""CPU smoke exports."""

from __future__ import annotations

from typing import Any

from ltx_trainer.dilated_sym_diff.paper import paper_card
from ltx_trainer.dilated_sym_diff.pipeline import run_demo


def evaluation_smoke() -> dict[str, Any]:
    demo = run_demo()
    summary = demo["summary"]
    return {
        "paper": paper_card(),
        "demo": demo,
        "IoU_anchor": summary["IoU_at_r8"],
        "status": "ok"
        if demo["IoU_r8"] >= 0.95
        and summary["delta_align_fig1"] >= 7.0
        and demo["chosen_r"] is not None
        else "fail",
    }
