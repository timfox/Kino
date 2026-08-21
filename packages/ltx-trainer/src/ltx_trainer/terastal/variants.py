"""Layer variants via S2D/D2S (Section III, Fig. 1)."""

from __future__ import annotations

import math
from typing import Any


def variant_transform_card(gamma: int = 2) -> dict[str, Any]:
    return {
        "gamma": gamma,
        "d2s": f"unfold channels → spatial dims (γ={gamma})",
        "s2d": f"fold spatial dims → channels (γ={gamma})",
        "weight_reduction_factor": 1.0 / (gamma**4),
        "purpose": "reshape computation for non-preferred OS/WS dataflow",
    }


def variant_latency_us(original_os_us: float, gamma: int = 2) -> float:
    """Stub: variant reduces OS latency toward WS level (Fig. 3)."""
    return original_os_us / (gamma * gamma)


def variant_accuracy_loss_pct(gamma: int = 2) -> float:
    """Fig. 3: individual variants 7–17% loss; paper uses min γ meeting latency target."""
    return 7.0 + 2.5 * (gamma - 2)


def minimum_gamma_for_latency(
    preferred_us: float,
    non_preferred_us: float,
    *,
    max_gamma: int = 4,
) -> int | None:
    """Minimum integer γ reducing non-preferred latency to ≤ preferred."""
    for g in range(2, max_gamma + 1):
        if variant_latency_us(non_preferred_us, g) <= preferred_us:
            return g
    return None


def valid_variant_combinations(
    n_variant_layers: int,
    accuracy_threshold: float,
    *,
    baseline_accuracy: float = 1.0,
) -> list[dict[str, Any]]:
    """Offline filter: combinations whose accuracy ≥ θ_m (Section IV-B)."""
    combos = []
    for k in range(0, n_variant_layers + 1):
        # Mean accuracy degrades with variant count; range reflects layer sensitivity
        mean_loss = 0.01 * k + 0.005 * k * k
        min_loss = mean_loss * 1.5
        max_loss = mean_loss * 0.5
        acc_mean = baseline_accuracy * (1.0 - mean_loss)
        if acc_mean >= accuracy_threshold * baseline_accuracy:
            combos.append(
                {
                    "n_variants_applied": k,
                    "accuracy_mean": round(acc_mean, 4),
                    "accuracy_min": round(baseline_accuracy * (1.0 - min_loss), 4),
                    "accuracy_max": round(baseline_accuracy * (1.0 - max_loss), 4),
                }
            )
    return combos
