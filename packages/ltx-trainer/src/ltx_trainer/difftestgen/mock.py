"""Mock exports for paper stub validation."""

from __future__ import annotations

from ltx_trainer.difftestgen.benchmarks import TABLE_TESTORA_OVERALL
from ltx_trainer.difftestgen.pipeline import evaluation_demo, evaluation_smoke_json


def mock_table_testora() -> dict:
    return dict(TABLE_TESTORA_OVERALL)


def mock_evaluation() -> dict:
    return evaluation_demo()


def evaluation_smoke() -> dict[str, bool]:
    return evaluation_smoke_json()
