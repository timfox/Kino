"""Paper table stubs for run_paper_stub_smoke."""

from __future__ import annotations

from ltx_trainer.fairlend.benchmarks import (
    TABLE_2_INCOME_QUARTILES,
    TABLE_3_BINNING,
    TABLE_5_TOP_DENIAL_RULES,
    TABLE_8_DIR_FINDINGS,
    TABLE_9_DIR_SUMMARY,
)
from ltx_trainer.fairlend.pipeline import evaluation_smoke


def table_2_income() -> list[dict[str, object]]:
    return TABLE_2_INCOME_QUARTILES


def table_3() -> list[dict[str, object]]:
    return TABLE_3_BINNING


def table_5() -> list[dict[str, object]]:
    return TABLE_5_TOP_DENIAL_RULES


def table_8() -> list[dict[str, object]]:
    return TABLE_8_DIR_FINDINGS


def table_9() -> list[dict[str, object]]:
    return TABLE_9_DIR_SUMMARY


def evaluation_smoke_export() -> dict[str, bool]:
    return evaluation_smoke()
