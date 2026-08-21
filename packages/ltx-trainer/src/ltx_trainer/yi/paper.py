"""Paper table stubs for run_paper_stub_smoke."""

from __future__ import annotations

from ltx_trainer.yi.benchmarks import (
    TABLE_1_UPDATE_SUPPORT,
    TABLE_BREAKDOWN,
    TABLE_DEEP100M,
    TABLE_SIFT800M,
)
from ltx_trainer.yi.pipeline import evaluation_smoke_json


def table_1() -> list[dict]:
    return TABLE_1_UPDATE_SUPPORT


def table_deep100m() -> list[dict]:
    return TABLE_DEEP100M


def table_sift800m() -> list[dict]:
    return TABLE_SIFT800M


def table_breakdown() -> list[dict]:
    return TABLE_BREAKDOWN


def evaluation_smoke_export() -> dict[str, bool]:
    return evaluation_smoke_json()
