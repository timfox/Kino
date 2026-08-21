"""Paper table stubs for run_paper_stub_smoke."""

from __future__ import annotations

from ltx_trainer.dsworld.benchmarks import (
    TABLE_1_AVG,
    TABLE_1_DSWORLD_ROW,
    TABLE_2_TRAINING,
    TABLE_3_AIDE_QWEN,
)
from ltx_trainer.dsworld.pipeline import evaluation_smoke_json


def table_1() -> dict:
    return dict(TABLE_1_AVG)


def table_1_row() -> dict:
    return dict(TABLE_1_DSWORLD_ROW)


def table_2() -> list[dict]:
    return TABLE_2_TRAINING


def table_3() -> list[dict]:
    return TABLE_3_AIDE_QWEN


def evaluation_smoke_export() -> dict[str, bool]:
    return evaluation_smoke_json()
