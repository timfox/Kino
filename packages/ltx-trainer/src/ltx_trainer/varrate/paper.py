"""Paper table stubs for VarRate."""

from __future__ import annotations

from ltx_trainer.varrate.baselines import (
    PAPER_ANCHORS,
    TABLE_1_TWO_MODEL,
    TABLE_2_REUSE,
    TABLE_3_PREFILL,
)
from ltx_trainer.varrate.pipeline import evaluation_smoke


def table_1() -> list[dict]:
    return list(TABLE_1_TWO_MODEL)


def table_2() -> list[dict]:
    return list(TABLE_2_REUSE)


def table_3() -> list[dict]:
    return list(TABLE_3_PREFILL)


def anchors() -> dict:
    return dict(PAPER_ANCHORS)


def evaluation_smoke_export() -> dict[str, bool]:
    return evaluation_smoke()
