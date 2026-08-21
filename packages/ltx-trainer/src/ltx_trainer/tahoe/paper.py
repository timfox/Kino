"""Paper table stubs for run_paper_stub_smoke."""

from __future__ import annotations

from ltx_trainer.tahoe.benchmarks import (
    TABLE_2_MAIN,
    TABLE_3_HELD_OUT,
    TABLE_4_ATTRIBUTION,
    TABLE_5_BY_CATEGORY,
)
from ltx_trainer.tahoe.pipeline import evaluation_smoke


def table_2() -> list[dict[str, object]]:
    return TABLE_2_MAIN


def table_3() -> list[dict[str, object]]:
    return TABLE_3_HELD_OUT


def table_4() -> list[dict[str, object]]:
    return TABLE_4_ATTRIBUTION


def table_5() -> list[dict[str, object]]:
    return TABLE_5_BY_CATEGORY


def evaluation_smoke_export() -> dict[str, bool]:
    return evaluation_smoke()
