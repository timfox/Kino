"""ID/OOD robustness metrics (Sec. 4.1)."""

from __future__ import annotations


def performance_drop(acc_id: float, acc_ood: float) -> float:
    """PD = (Acc_ID − Acc_OOD) / Acc_ID (ratio, not percent)."""
    if acc_id <= 0:
        return 0.0
    return (acc_id - acc_ood) / acc_id


def harmonic_mean(acc_id: float, acc_ood: float) -> float:
    """H = 2 Acc_ID Acc_OOD / (Acc_ID + Acc_OOD) (percent scale)."""
    denom = acc_id + acc_ood
    if denom <= 0:
        return 0.0
    return 2.0 * acc_id * acc_ood / denom
