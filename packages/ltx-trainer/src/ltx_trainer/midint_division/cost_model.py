"""Multiplication-count cost model (§2.3): 5–7 full classical mults per division."""

from __future__ import annotations

from typing import Any

from ltx_trainer.midint_division.constants import FULL_MULT_COST_MAX, FULL_MULT_COST_MIN


def full_mult_bounds() -> dict[str, int]:
    return {"min": FULL_MULT_COST_MIN, "max": FULL_MULT_COST_MAX}


def estimate_full_mults(
    *,
    h: int,
    k: int,
    refine_loops: int,
) -> dict[str, Any]:
    """Heuristic count aligned with §2.3 (shinv 2–4 + quotient stage 2–3 full mults)."""
    shinv_min = 2
    shinv_max = 4
    quotient_stage = 3  # u·shinv double-width (2) + v·q (1)
    # Rare extra Refine loop full mult in PowDiff when floor(h/2)-1 < ell_{i+1} < ceil(h/2)
    extra = 1 if refine_loops > 3 and k > h // 2 else 0
    total_min = shinv_min + quotient_stage
    total_max = shinv_max + quotient_stage + extra
    return {
        "shinv_full_mult_min": shinv_min,
        "shinv_full_mult_max": shinv_max,
        "quotient_stage_full_mult": quotient_stage,
        "total_min": max(FULL_MULT_COST_MIN, total_min),
        "total_max": min(FULL_MULT_COST_MAX, total_max),
        "paper_bounds": full_mult_bounds(),
    }


def div_to_mul_ratio(div_ms: float, mul_ms: float) -> float:
    if mul_ms <= 0:
        return 0.0
    return round(div_ms / mul_ms, 2)
