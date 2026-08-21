"""Token-budget decomposition and binding gain helpers (Eq. 1)."""

from __future__ import annotations


def decompose_total_gain(
    bl256: float,
    bl1024: float,
    cot1024: float,
) -> dict[str, float]:
    """∆Total = [BL1024 − BL256]_budget + [CoT − BL1024]_instruction."""
    budget = bl1024 - bl256
    instruction = cot1024 - bl1024
    return {
        "baseline_256": bl256,
        "baseline_1024": bl1024,
        "cot_1024": cot1024,
        "budget_delta_pp": budget,
        "instruction_delta_pp": instruction,
        "total_delta_pp": cot1024 - bl256,
    }


def s2t_gap(s2t: float, t2t: float) -> float:
    """Speech minus text accuracy (percentage points); negative = speech worse."""
    return s2t - t2t
