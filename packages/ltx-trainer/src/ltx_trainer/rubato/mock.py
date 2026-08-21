"""Rubato InterMo smoke."""

from __future__ import annotations

from typing import Any


def toy_measure_valid() -> tuple[str, list[str]]:
    return "|3/4k-4", ["1/4", "1/4", "1/4"]


def toy_measure_invalid_sum() -> tuple[str, list[str]]:
    return "|3/4k-4", ["1/4", "1/8"]


def evaluation_smoke() -> dict[str, Any]:
    from ltx_trainer.rubato.intermo import validate_measure_metric_sum

    barline, intervals = toy_measure_valid()
    ok = validate_measure_metric_sum(barline, intervals)
    bad_barline, bad_intervals = toy_measure_invalid_sum()
    bad_ok = validate_measure_metric_sum(bad_barline, bad_intervals)
    return {"valid_measure_ok": ok, "invalid_measure_ok": bad_ok}
