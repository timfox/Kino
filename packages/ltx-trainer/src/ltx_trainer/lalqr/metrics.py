"""Paper benchmark anchors (Tables 1–2) and projection bounds (Lemma 4.1)."""

from __future__ import annotations

from typing import Any

import numpy as np


def projection_calibrated_setpoint(
    beta_x_star: float,
    *,
    rho: float,
    eta_bar: float = 0.0,
) -> float:
    """β^{z,*} = (β^{x,*} − η̄) / √ρ — Eq. (21)."""
    gamma = max(float(np.sqrt(max(rho, 1e-12))), 1e-8)
    return (beta_x_star - eta_bar) / gamma


def raw_latent_tracking_bound(
    beta_z: float,
    beta_z_star: float,
    *,
    rho: float,
    eta: float = 0.0,
    eta_bar: float = 0.0,
) -> float:
    """|β^x − β^{x,*}| bound from Lemma 4.1."""
    gamma = float(np.sqrt(max(rho, 1e-12)))
    return gamma * abs(beta_z - beta_z_star) + abs(eta - eta_bar)


def table1_t2vsafetybench() -> list[dict[str, Any]]:
    """Violation rate ↓ on Wan — Table 1 (paper anchors)."""
    categories = (
        "Copyright & Trademarks",
        "Pornography",
        "Gore",
        "Public Figure",
        "Sequential Action Risk",
    )
    wan = (0.710, 0.500, 0.420, 0.105, 0.109)
    safree = (0.515, 0.365, 0.275, 0.060, 0.091)
    unlearning = (0.710, 0.500, 0.395, 0.100, 0.109)
    slider = (0.650, 0.475, 0.310, 0.050, 0.109)
    ours = (0.370, 0.095, 0.140, 0.030, 0.073)
    rows: list[dict[str, Any]] = []
    for i, cat in enumerate(categories):
        rows.extend(
            [
                {"benchmark": "T2VSafetyBench", "metric": "violation_rate", "category": cat, "method": "Wan", "value": wan[i]},
                {"benchmark": "T2VSafetyBench", "metric": "violation_rate", "category": cat, "method": "SAFREE", "value": safree[i]},
                {"benchmark": "T2VSafetyBench", "metric": "violation_rate", "category": cat, "method": "Unlearning", "value": unlearning[i]},
                {"benchmark": "T2VSafetyBench", "metric": "violation_rate", "category": cat, "method": "Slider", "value": slider[i]},
                {"benchmark": "T2VSafetyBench", "metric": "violation_rate", "category": cat, "method": "LA-LQR", "value": ours[i]},
            ]
        )
    return rows


def table1_vbench_subject() -> list[dict[str, Any]]:
    categories = (
        "Copyright & Trademarks",
        "Pornography",
        "Gore",
        "Public Figure",
        "Sequential Action Risk",
    )
    ours = (0.976, 0.974, 0.974, 0.970, 0.962)
    return [
        {
            "benchmark": "T2VSafetyBench",
            "metric": "vbench_subject_consistency",
            "category": cat,
            "method": "LA-LQR",
            "value": v,
        }
        for cat, v in zip(categories, ours, strict=True)
    ]


def table2_safesora() -> list[dict[str, Any]]:
    categories = ("Violence", "Terrorism", "Racism", "Sexual", "Animal Abuse")
    hunyuan = (0.320, 0.320, 0.125, 0.438, 0.370)
    ours = (0.222, 0.240, 0.000, 0.000, 0.148)
    rows: list[dict[str, Any]] = []
    for cat, base, steered in zip(categories, hunyuan, ours, strict=True):
        rows.append(
            {
                "benchmark": "SafeSora",
                "metric": "violation_rate",
                "category": cat,
                "method": "Hunyuan",
                "value": base,
            }
        )
        rows.append(
            {
                "benchmark": "SafeSora",
                "metric": "violation_rate",
                "category": cat,
                "method": "LA-LQR",
                "value": steered,
            }
        )
    return rows


def average_violation(rows: list[dict[str, Any]], *, method: str, benchmark: str) -> float:
    vals = [
        float(r["value"])
        for r in rows
        if r.get("method") == method and r.get("benchmark") == benchmark and r.get("metric") == "violation_rate"
    ]
    return float(np.mean(vals)) if vals else 0.0
