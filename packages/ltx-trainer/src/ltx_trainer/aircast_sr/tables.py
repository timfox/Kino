"""Paper Tables I–VIII (AirCast-SR, arXiv:2605.26130)."""

from __future__ import annotations

from typing import Any


def table_i_precipitation() -> list[dict[str, Any]]:
    return [
        {"case": "Dec 2022", "lead": "6h", "EM": {"r": 0.40, "rmse": 0.87, "bias": 0.00}, "HRRR": {"r": 0.64}, "GC": {"r": 0.75}},
        {"case": "Jun 2022", "lead": "18h", "EM": {"r": 0.43, "rmse": 3.03, "bias": 0.00}, "HRRR": {"r": 0.26}, "GC": {"r": 0.55}},
        {"case": "Mar 2023", "lead": "48h", "EM": {"r": 0.24, "rmse": 1.89, "bias": 0.01}, "HRRR": {"r": 0.27}, "GC": {"r": 0.43}},
    ]


def table_ii_temperature() -> list[dict[str, Any]]:
    return [
        {"case": "Dec 2022", "lead": "6h", "EM": {"r": 0.9716, "bias_k": -0.01}, "HRRR": {"r": 0.9951}},
        {"case": "Jun 2022", "lead": "48h", "EM": {"r": 0.8331, "bias_k": 0.01}, "HRRR": {"r": 0.9882}},
        {"case": "Mar 2023", "lead": "24h", "EM": {"r": 0.9320, "bias_k": 0.00}, "HRRR": {"r": 0.9920}},
    ]


def table_iii_surface_pressure() -> list[dict[str, Any]]:
    return [
        {"case": "Dec 2022", "lead": "6h", "EM": {"r": 0.9666, "bias_pa": -5.8}, "HRRR": {"r": 0.9890, "bias_pa": 15.0}},
        {"case": "Mar 2023", "lead": "6h", "EM": {"r": 0.9738, "bias_pa": 0.3}, "HRRR": {"r": 0.9901, "bias_pa": 11.7}},
    ]


def table_iv_humidity() -> list[dict[str, Any]]:
    return [
        {"case": "Dec 2022", "lead": "6h", "EM": {"r": 0.9006, "bias_1e3": 0.026}, "HRRR": {"r": 0.9915}},
    ]


def table_v_longwave() -> list[dict[str, Any]]:
    return [
        {"case": "Dec 2022", "lead": "2h", "EM": {"r": 0.8083, "rmse": 28.8}, "HRRR": {"r": 0.8230, "rmse": 36.0}},
    ]


def table_vi_u_wind() -> list[dict[str, Any]]:
    return [
        {"case": "Dec 2022", "lead": "24h", "EM": {"r": 0.7930, "bias_ms": 0.00}, "HRRR": {"r": 0.9426}, "GC": {"r": 0.8462, "bias_ms": 0.76}},
    ]


def table_vii_v_wind() -> list[dict[str, Any]]:
    return [
        {"case": "Dec 2022", "lead": "6h", "EM": {"r": 0.7469, "bias_ms": 0.00}, "HRRR": {"r": 0.9261}, "GC": {"r": 0.5627, "bias_ms": 1.94}},
    ]


def table_viii_zero_shot_t2m() -> list[dict[str, Any]]:
    return [
        {"domain": "India", "lead": "6h", "EM": {"r": 0.88, "rmse_k": 3.74}, "GC": {"r": 0.95, "rmse_k": 2.85}},
        {"domain": "India", "lead": "48h", "EM": {"r": 0.89, "rmse_k": 3.29}, "GC": {"r": 0.94}},
        {"domain": "Germany", "lead": "6h", "EM": {"r": 0.63, "rmse_k": 1.54}, "GC": {"r": 0.78}},
        {"domain": "Germany", "lead": "48h", "EM": {"r": 0.37, "rmse_k": 1.80}, "GC": {"r": 0.66}},
    ]
