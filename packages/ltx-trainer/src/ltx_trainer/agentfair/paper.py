"""Paper table stubs for run_paper_stub_smoke."""

from __future__ import annotations

from ltx_trainer.agentfair.benchmarks import (
    TABLE_4_CASE_STUDIES,
    TABLE_6_SPEARMAN,
    TABLE_7_EFFICIENCY,
    TABLE_8_CRITIC_ABLATION,
)
from ltx_trainer.agentfair.pipeline import evaluation_smoke_fix


def table_4() -> list[dict]:
    return TABLE_4_CASE_STUDIES


def table_6() -> list[dict]:
    return TABLE_6_SPEARMAN


def table_7() -> dict:
    return TABLE_7_EFFICIENCY


def table_8() -> list[dict]:
    return TABLE_8_CRITIC_ABLATION


def evaluation_smoke_export() -> dict[str, bool]:
    return evaluation_smoke_fix()
