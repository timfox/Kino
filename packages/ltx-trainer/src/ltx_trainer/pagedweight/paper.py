"""Paper table stubs for run_paper_stub_smoke."""

from __future__ import annotations

from ltx_trainer.pagedweight.benchmarks import (
    TABLE_1_MODELS,
    TABLE_2_LONGBENCH,
    TABLE_3_THROUGHPUT,
    TABLE_4_ABLATION,
)
from ltx_trainer.pagedweight.pipeline import evaluation_smoke_json


def table_1() -> list[dict]:
    return TABLE_1_MODELS


def table_2() -> list[dict]:
    return TABLE_2_LONGBENCH


def table_3() -> list[dict]:
    return TABLE_3_THROUGHPUT


def table_4() -> list[dict]:
    return TABLE_4_ABLATION


def evaluation_smoke_export() -> dict[str, bool]:
    return evaluation_smoke_json()
