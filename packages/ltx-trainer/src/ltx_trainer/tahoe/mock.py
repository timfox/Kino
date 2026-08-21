"""Mock exports for paper stub validation."""

from __future__ import annotations

from ltx_trainer.tahoe.benchmarks import TABLE_2_MAIN
from ltx_trainer.tahoe.pipeline import evaluation_demo, evaluation_smoke


def mock_table_2() -> list[dict[str, object]]:
    return list(TABLE_2_MAIN)


def mock_evaluation() -> dict[str, object]:
    return evaluation_demo()


def mock_smoke() -> dict[str, bool]:
    return evaluation_smoke()


def evaluation_smoke() -> dict[str, bool]:
    """Alias for run_paper_stub_smoke discovery."""
    from ltx_trainer.tahoe.pipeline import evaluation_smoke as pipeline_smoke

    return pipeline_smoke()
