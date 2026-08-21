"""Paper table stubs for run_paper_stub_smoke (CAPC)."""

from __future__ import annotations

from ltx_trainer.capc.baselines import (
    LONGBENCH_DOMINANCE,
    LONGBENCH_MEAN_ROW,
    PAPER_ANCHORS,
    RHO_CROSS_TABLE,
    TAU_BENCH_RETAIL,
)
from ltx_trainer.capc.pipeline import evaluation_smoke


def table_longbench() -> dict:
    return dict(LONGBENCH_DOMINANCE)


def table_longbench_mean() -> dict:
    return dict(LONGBENCH_MEAN_ROW)


def table_rho_cross() -> dict:
    return dict(RHO_CROSS_TABLE)


def table_tau() -> dict:
    return dict(TAU_BENCH_RETAIL)


def anchors() -> dict:
    return dict(PAPER_ANCHORS)


def evaluation_smoke_export() -> dict[str, bool]:
    return evaluation_smoke()
