"""CPU smoke exports."""

from __future__ import annotations

from typing import Any

from ltx_trainer.ai_hpc_workflows.constants import TIPS
from ltx_trainer.ai_hpc_workflows.paper import paper_card
from ltx_trainer.ai_hpc_workflows.pipeline import run_demo
from ltx_trainer.ai_hpc_workflows.tips import checklist_score


def evaluation_smoke() -> dict[str, Any]:
    demo = run_demo()
    return {
        "paper": paper_card(),
        "demo": demo,
        "checklist": checklist_score({s: True for _, s, _ in TIPS}),
        "status": "ok",
    }
