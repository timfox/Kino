"""CPU smoke exports."""

from __future__ import annotations

from typing import Any

from ltx_trainer.pccl.paper import paper_card
from ltx_trainer.pccl.pipeline import run_demo


def evaluation_smoke() -> dict[str, Any]:
    demo = run_demo()
    return {
        "paper": paper_card(),
        "demo": demo,
        "synthesis_ok": demo["all_gather_pg"]["n_conditions"] == 3,
        "status": "ok",
    }
