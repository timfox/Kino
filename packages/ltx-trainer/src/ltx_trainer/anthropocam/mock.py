"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.anthropocam.config import OPTIMAL_STYLE_WEIGHT, PAPER_ARXIV
from ltx_trainer.anthropocam.pipeline import evaluation_demo, framework_card


def evaluation_smoke() -> dict[str, Any]:
    demo = evaluation_demo()
    fw = demo["framework"]
    assert fw["paper"] == PAPER_ARXIV
    assert fw["optimal_style_weight"] == OPTIMAL_STYLE_WEIGHT
    assert demo["forward"]["losses"]["total"] >= 0.0
    optimal = next(r for r in demo["style_weight_sensitivity"] if r["label"] == "optimal")
    assert optimal["w_s"] == 5
    mobile = next(r for r in demo["resolution_latency"] if r["mobile_ok"])
    assert mobile["width"] == 1280
    return {"status": "ok", "paper": fw["paper"], "style_weight": fw["optimal_style_weight"]}


__all__ = ["evaluation_smoke", "framework_card"]
