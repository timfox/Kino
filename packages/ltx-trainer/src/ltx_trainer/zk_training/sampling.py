"""Interactive sampling statistics — Table 6 / Sec. B.4."""

from __future__ import annotations

import math


def sample_count(*, miss_prob: float, deviation_fraction: float) -> int:
    """k ≥ ln(1/τ) / f for target miss probability τ at deviation fraction f (Sec. B.4.1)."""
    if deviation_fraction <= 0:
        raise ValueError("deviation_fraction must be positive")
    # Paper Table 6 uses the small-f approximation with integer truncation (e.g. k=4605 at τ=10^-20, f=1%).
    k = math.log(1.0 / miss_prob) / deviation_fraction
    return max(1, int(k))


def miss_probability(k: int, deviation_fraction: float) -> float:
    return (1.0 - deviation_fraction) ** k


def table6_sample_counts() -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for target, label in ((1e-20, "10^-20"), (2 ** -128, "2^-128")):
        for frac, frac_label in ((0.10, "10%"), (0.01, "1%"), (0.001, "0.1%")):
            rows.append(
                {
                    "target": label,
                    "deviation_fraction": frac_label,
                    "k": sample_count(miss_prob=target, deviation_fraction=frac),
                }
            )
    return rows
