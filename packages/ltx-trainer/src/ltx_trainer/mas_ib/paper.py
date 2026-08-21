"""Paper table stubs for run_paper_stub_smoke."""

from __future__ import annotations

from ltx_trainer.mas_ib.benchmarks import (
    TABLE_1_DELTA_REGIMES,
    TABLE_2_GAINS,
    TABLE_6_ALFWORLD_ZEROSHOT,
)
from ltx_trainer.mas_ib.pipeline import evaluation_smoke_json


def table_1() -> list[dict]:
    return TABLE_1_DELTA_REGIMES


def table_2() -> list[dict]:
    return TABLE_2_GAINS


def table_6() -> list[dict]:
    return TABLE_6_ALFWORLD_ZEROSHOT


def evaluation_smoke_export() -> dict[str, bool]:
    return evaluation_smoke_json()
