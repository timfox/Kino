"""Paper table stubs for run_paper_stub_smoke."""

from __future__ import annotations

from ltx_trainer.neurowl.benchmarks import (
    TABLE_1_STATS,
    TABLE_2_NEUROWL_ONT,
    TABLE_4_ABLATION_FOODONA_FULL,
)
from ltx_trainer.neurowl.pipeline import evaluation_smoke_fix


def table_1() -> list[dict]:
    return TABLE_1_STATS


def table_2() -> list[dict]:
    return TABLE_2_NEUROWL_ONT


def table_4() -> dict:
    return TABLE_4_ABLATION_FOODONA_FULL


def evaluation_smoke_export() -> dict[str, bool]:
    return evaluation_smoke_fix()
