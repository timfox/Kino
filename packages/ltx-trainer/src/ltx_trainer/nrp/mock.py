"""Mock exports for paper stub validation."""

from __future__ import annotations

from ltx_trainer.nrp.benchmarks import TABLE_I
from ltx_trainer.nrp.pipeline import evaluation_demo, evaluation_smoke


def mock_table_i() -> list[dict[str, str]]:
    return list(TABLE_I)


def mock_evaluation() -> dict[str, object]:
    return evaluation_demo()


def mock_smoke() -> dict[str, bool]:
    return evaluation_smoke()
