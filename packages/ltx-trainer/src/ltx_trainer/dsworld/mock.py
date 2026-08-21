"""Mock exports for paper stub validation."""

from __future__ import annotations

from ltx_trainer.dsworld.benchmarks import TABLE_1_AVG
from ltx_trainer.dsworld.pipeline import evaluation_demo, evaluation_smoke_json


def mock_table_1() -> dict:
    return dict(TABLE_1_AVG)


def mock_evaluation() -> dict:
    return evaluation_demo()


def evaluation_smoke() -> dict[str, bool]:
    return evaluation_smoke_json()
