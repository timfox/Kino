"""Mock exports for paper stub validation."""

from __future__ import annotations

from ltx_trainer.mas_ib.benchmarks import TABLE_2_GAINS
from ltx_trainer.mas_ib.pipeline import evaluation_demo, evaluation_smoke_json


def mock_table_2() -> list[dict]:
    return list(TABLE_2_GAINS)


def mock_evaluation() -> dict:
    return evaluation_demo()


def evaluation_smoke() -> dict[str, bool]:
    return evaluation_smoke_json()
