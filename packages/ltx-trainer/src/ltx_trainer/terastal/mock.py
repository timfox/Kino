"""CPU smoke exports."""

from __future__ import annotations

from typing import Any

from ltx_trainer.terastal.constants import MISS_RATE_REDUCTION_VS
from ltx_trainer.terastal.paper import paper_card
from ltx_trainer.terastal.pipeline import run_demo


def evaluation_smoke() -> dict[str, Any]:
    demo = run_demo()
    return {
        "paper": paper_card(),
        "demo": demo,
        "miss_reduction_ok": demo["miss_rate_stub"]["reduction_vs_fcfs"] >= MISS_RATE_REDUCTION_VS["fcfs"] - 0.01,
        "schedule_ok": demo["schedule_round"]["completed"] >= 1,
        "status": "ok",
    }
