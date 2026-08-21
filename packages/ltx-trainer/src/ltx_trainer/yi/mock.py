"""Mock exports for paper stub validation."""

from __future__ import annotations

from ltx_trainer.yi.benchmarks import TABLE_DEEP100M
from ltx_trainer.yi.pipeline import evaluation_demo, evaluation_smoke_json


def mock_table_deep100m() -> list[dict]:
    return list(TABLE_DEEP100M)


def mock_evaluation() -> dict:
    return evaluation_demo()


def evaluation_smoke() -> dict[str, bool]:
    return evaluation_smoke_json()
