"""Tetris tile packing smoke (arXiv:2605.25538)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.smoke_util import load_sibling


def evaluation_smoke() -> dict[str, Any]:
    cfg_mod = load_sibling(__file__, "config")
    cfg = cfg_mod.TetrisConfig()
    step: dict[str, float] = {"num_canvases": 1.0, "detector_calls_saved": 0.0}
    try:
        pipe = load_sibling(__file__, "pipeline")
        step = pipe.training_step_demo(cfg)
    except ImportError:
        pass
    return {
        "paper": getattr(cfg, "paper_arxiv", "arXiv:2605.25538"),
        "num_canvases": step.get("num_canvases", 1.0),
        "detector_calls_saved": step.get("detector_calls_saved", 0.0),
        "relevance_threshold": cfg.relevance_threshold,
    }
