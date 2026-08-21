"""Paper table stubs for run_paper_stub_smoke."""

from __future__ import annotations

from ltx_trainer.nrp.benchmarks import TABLE_I
from ltx_trainer.nrp.pipeline import evaluation_smoke


def table_i() -> list[dict[str, str]]:
    return TABLE_I


def evaluation_smoke_export() -> dict[str, bool]:
    return evaluation_smoke()
