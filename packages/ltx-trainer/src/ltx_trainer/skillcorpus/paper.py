"""Paper table stubs for run_paper_stub_smoke."""

from __future__ import annotations

from ltx_trainer.skillcorpus.benchmarks import (
    TABLE_1_CELLS,
    TABLE_1_POOLED_DELTA,
    TABLE_2_ABLATION,
    TABLE_7_RETRIEVAL,
)
from ltx_trainer.skillcorpus.pipeline import evaluation_smoke_json


def table_1() -> list[dict]:
    return list(TABLE_1_CELLS)


def table_1_pooled() -> dict:
    return dict(TABLE_1_POOLED_DELTA)


def table_2() -> list[dict]:
    return list(TABLE_2_ABLATION)


def table_7() -> list[dict]:
    return list(TABLE_7_RETRIEVAL)


def evaluation_smoke_export() -> dict[str, bool]:
    return evaluation_smoke_json()
