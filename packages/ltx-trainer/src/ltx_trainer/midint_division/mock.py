"""CPU smoke exports."""

from __future__ import annotations

from typing import Any

from ltx_trainer.midint_division.paper import paper_card
from ltx_trainer.midint_division.pipeline import run_demo


def evaluation_smoke() -> dict[str, Any]:
    demo = run_demo()
    return {
        "paper": paper_card(),
        "demo": demo,
        "examples_ok": demo["all_examples_ok"],
        "div_mul_near_optimal": demo["table_1_row"]["div_over_mul"] <= 5.5,
        "status": "ok" if demo["all_examples_ok"] else "fail",
    }
