"""CPU smoke exports."""

from __future__ import annotations

from typing import Any

from ltx_trainer.set_cuda_graph.paper import paper_card
from ltx_trainer.set_cuda_graph.pipeline import run_demo


def evaluation_smoke() -> dict[str, Any]:
    demo = run_demo()
    return {
        "paper": paper_card(),
        "demo": demo,
        "set_beats_batching": demo["set_beats_batching"],
        "status": "ok" if demo["set_beats_batching"] and demo["dispatch_demo"]["jobs_completed"] > 0 else "fail",
    }
