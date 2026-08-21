"""Mock exports for paper stub validation."""

from __future__ import annotations

from ltx_trainer.neurowl.benchmarks import TABLE_2_NEUROWL_ONT
from ltx_trainer.neurowl.pipeline import evaluation_demo, evaluation_smoke_fix


def mock_table_2() -> list[dict]:
    return list(TABLE_2_NEUROWL_ONT)


def mock_evaluation() -> dict:
    return evaluation_demo()


def evaluation_smoke() -> dict[str, bool]:
    return evaluation_smoke_fix()
