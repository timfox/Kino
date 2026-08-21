"""Fig. 6.2 and example anchors."""

from __future__ import annotations

from typing import Any

from ltx_trainer.dynamic_gp.constants import HEAT_ERROR_BY_M, WAVE_EXAMPLE


def heat_error_curve() -> list[dict[str, int | float]]:
    return [{"M": m, "error_l2": err} for m, err in HEAT_ERROR_BY_M]


def wave_example_card() -> dict[str, Any]:
    return dict(WAVE_EXAMPLE)


def summary_anchors() -> dict[str, Any]:
    m3, m101 = HEAT_ERROR_BY_M[0][1], HEAT_ERROR_BY_M[-1][1]
    return {
        "heat_error_M3": m3,
        "heat_error_M101": m101,
        "error_reduction_M3_to_M101": round(m3 / m101, 2),
        "wave_M_fourier": WAVE_EXAMPLE["M_fourier"],
        "wave_D": WAVE_EXAMPLE["D_state"],
    }
