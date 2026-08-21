"""Mock exports for paper stub validation."""

from __future__ import annotations

from ltx_trainer.pagedweight.benchmarks import TABLE_4_ABLATION
from ltx_trainer.pagedweight.pipeline import evaluation_demo, evaluation_smoke_json


def mock_table_4() -> list[dict]:
    return list(TABLE_4_ABLATION)


def mock_evaluation() -> dict:
    return evaluation_demo()


def evaluation_smoke() -> dict[str, bool]:
    return evaluation_smoke_json()
