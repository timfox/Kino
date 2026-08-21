"""Paper table stubs for run_paper_stub_smoke."""

from __future__ import annotations

from ltx_trainer.difftestgen.benchmarks import (
    TABLE_CHACO,
    TABLE_COST,
    TABLE_GENERATED_TESTS,
    TABLE_TESTORA_OVERALL,
)
from ltx_trainer.difftestgen.pipeline import evaluation_smoke_json


def table_1() -> dict:
    """Testora-data overall comparison (Fig. 5)."""
    return dict(TABLE_TESTORA_OVERALL)


def table_2() -> list[dict]:
    return TABLE_CHACO


def table_3() -> list[dict]:
    return TABLE_COST


def table_4() -> list[dict]:
    return TABLE_GENERATED_TESTS


def evaluation_smoke_export() -> dict[str, bool]:
    return evaluation_smoke_json()
