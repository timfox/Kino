"""Paper table stubs for run_paper_stub_smoke."""

from __future__ import annotations

from ltx_trainer.trajgenagent.benchmarks import (
    TABLE_II_NUMOSIM,
    TABLE_III_ANOMALY,
    TABLE_VI_KINEMATICS,
    TABLE_VII_KINEMATICS_ANOMALY,
    TABLE_VIII_TEMPERATURE,
)
from ltx_trainer.trajgenagent.pipeline import evaluation_smoke


def table_ii_numosim() -> list[dict[str, object]]:
    return TABLE_II_NUMOSIM


def table_iii_anomaly() -> list[dict[str, object]]:
    return TABLE_III_ANOMALY


def table_vi_kinematics() -> list[dict[str, object]]:
    return TABLE_VI_KINEMATICS


def table_vii_kinematics_anomaly() -> list[dict[str, object]]:
    return TABLE_VII_KINEMATICS_ANOMALY


def table_viii_temperature() -> list[dict[str, object]]:
    return TABLE_VIII_TEMPERATURE


def evaluation_smoke_export() -> dict[str, bool]:
    return evaluation_smoke()
