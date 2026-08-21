"""Mock exports for paper stub validation."""

from __future__ import annotations

from ltx_trainer.fairlend.benchmarks import TABLE_3_BINNING
from ltx_trainer.fairlend.pipeline import evaluation_demo, evaluation_smoke as pipeline_smoke


def mock_table_3() -> list[dict[str, object]]:
    return list(TABLE_3_BINNING)


def mock_evaluation() -> dict[str, object]:
    return evaluation_demo()


def mock_smoke() -> dict[str, bool]:
    return pipeline_smoke()


def evaluation_smoke() -> dict[str, bool]:
    return pipeline_smoke()
