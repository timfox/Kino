"""Temperature-scaled dataset interleaving (Eq. 1, T=2)."""

from __future__ import annotations

import math
from typing import Any


def interleave_probabilities(counts: dict[str, int], temperature: float = 2.0) -> dict[str, float]:
    """pi = ni^(1/T) / sum_j nj^(1/T)."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    powered = {k: n ** (1.0 / temperature) for k, n in counts.items()}
    total = sum(powered.values())
    return {k: v / total for k, v in powered.items()}


def table1_distribution(temperature: float = 2.0) -> list[dict[str, Any]]:
    """Table 1 — training data counts and T=2 sampling probabilities."""
    counts = {
        "ASR": 19_248,
        "SQA": 474_888,
        "MC": 380_056,
        "SSUM": 35_748,
        "ST": 29_343,
        "AChap": 37_862,
        "Instruct": 71_013,
    }
    total = sum(counts.values())
    probs = interleave_probabilities(counts, temperature)
    rows = []
    for name, n in counts.items():
        rows.append(
            {
                "dataset": name,
                "samples": n,
                "init_pct": round(100.0 * n / total, 2),
                "sample_pct_T2": round(100.0 * probs[name], 2),
            }
        )
    rows.append({"dataset": "Total", "samples": total, "init_pct": 100.0, "sample_pct_T2": 100.0})
    return rows
