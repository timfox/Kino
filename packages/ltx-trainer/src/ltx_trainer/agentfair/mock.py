"""Mock exports for paper stub validation."""

from __future__ import annotations

from ltx_trainer.agentfair.benchmarks import TABLE_4_CASE_STUDIES, TABLE_6_SPEARMAN
from ltx_trainer.agentfair.pipeline import evaluation_demo, evaluation_smoke_fix


def mock_table_4() -> list[dict]:
    return list(TABLE_4_CASE_STUDIES)


def mock_table_6() -> list[dict]:
    return list(TABLE_6_SPEARMAN)


def mock_evaluation() -> dict:
    return evaluation_demo()


def mock_smoke() -> dict[str, bool]:
    return evaluation_smoke_fix()


def evaluation_smoke() -> dict[str, bool]:
    return evaluation_smoke_fix()
