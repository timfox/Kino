"""VarRate mock exports for paper stub validation."""

from __future__ import annotations

from ltx_trainer.varrate.baselines import PAPER_ANCHORS
from ltx_trainer.varrate.pipeline import evaluation_demo, evaluation_smoke as _smoke


def mock_anchors() -> dict:
    return dict(PAPER_ANCHORS)


def mock_evaluation() -> dict:
    return evaluation_demo()


def evaluation_smoke() -> dict[str, bool]:
    return _smoke()
