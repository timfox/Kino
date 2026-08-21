"""DeepSWE leaderboard, separation, and harness helpers."""

from __future__ import annotations

from typing import Any

from ltx_trainer.deepswe.benchmarks import (
    DEEPSWE_PASS_NORM,
    LEADERBOARD_PASS_RATE,
    SWE_BENCH_PRO_PASS,
)
from ltx_trainer.deepswe.constants import SCOPE_MULTIPLIER_VS_PRO


def deepswe_leaderboard() -> dict[str, float]:
    """Model id → pass rate (%) on DeepSWE publication snapshot."""
    out: dict[str, float] = {}
    for model, row in LEADERBOARD_PASS_RATE.items():
        key = model.replace("[", "_").replace("]", "").replace(".", "_").replace("-", "_")
        out[key] = float(row["pass_pct"])
    return out


def leaderboard_ranked() -> list[tuple[str, float]]:
    lb = deepswe_leaderboard()
    return sorted(lb.items(), key=lambda x: -x[1])


def separation_vs_swe_bench_pro() -> dict[str, Any]:
    """Pass-rate span on comparable models (blog: 70pp DeepSWE vs 30pp Pro)."""
    deepswe_vals = [v for v in DEEPSWE_PASS_NORM.values() if v > 0]
    pro_vals = list(SWE_BENCH_PRO_PASS.values())
    return {
        "deepswe_min": min(deepswe_vals) if deepswe_vals else 0.0,
        "deepswe_max": max(deepswe_vals) if deepswe_vals else 0.0,
        "deepswe_span_pp": (max(deepswe_vals) - min(deepswe_vals)) if deepswe_vals else 0.0,
        "swe_bench_pro_min": min(pro_vals),
        "swe_bench_pro_max": max(pro_vals),
        "swe_bench_pro_span_pp": max(pro_vals) - min(pro_vals),
        "scope_multiplier_lines_vs_pro": SCOPE_MULTIPLIER_VS_PRO,
    }


def model_delta_deepswe_minus_pro(model_key: str) -> float | None:
    """Per-model gap (DeepSWE − SWE-Bench Pro) where both exist."""
    d = DEEPSWE_PASS_NORM.get(model_key)
    p = SWE_BENCH_PRO_PASS.get(model_key)
    if d is None or p is None:
        return None
    return d - p
