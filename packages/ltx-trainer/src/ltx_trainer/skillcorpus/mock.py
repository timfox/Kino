"""Mock exports for paper stub validation."""

from __future__ import annotations

from ltx_trainer.skillcorpus.benchmarks import TABLE_1_POOLED_DELTA
from ltx_trainer.skillcorpus.pipeline import evaluation_demo, evaluation_smoke_json


def mock_table_1_pooled() -> dict:
    return dict(TABLE_1_POOLED_DELTA)


def mock_evaluation() -> dict:
    return evaluation_demo()


def evaluation_smoke() -> dict[str, bool]:
    return evaluation_smoke_json()
