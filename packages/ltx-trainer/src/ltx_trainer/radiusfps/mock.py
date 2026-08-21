"""CPU smoke exports."""

from __future__ import annotations

from typing import Any

from ltx_trainer.radiusfps.paper import paper_card
from ltx_trainer.radiusfps.pipeline import run_demo


def evaluation_smoke() -> dict[str, Any]:
    demo = run_demo()
    return {
        "paper": paper_card(),
        "demo": demo,
        "exact_fps_preserved": demo["exact_match"],
        "status": "ok" if demo["exact_match"] and demo["beats_gpu_fps_e2e"] else "fail",
    }
