"""Semantic Transmission Efficiency (STE) metric (Sec. V-A)."""

from __future__ import annotations


def semantic_transmission_efficiency(
    retention_per_client: list[float] | tuple[float, ...],
    uplink_latencies_s: list[float] | tuple[float, ...],
) -> float:
    """E = sum_m f_m(K_m) / max_m T^U_m (Eq. 20)."""
    if not retention_per_client:
        return 0.0
    num = float(sum(retention_per_client))
    den = float(max(uplink_latencies_s))
    if den <= 0:
        return 0.0
    return num / den
