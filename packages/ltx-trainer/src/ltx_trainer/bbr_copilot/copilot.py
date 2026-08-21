"""BBR-Copilot padding controller + data generator (§3.3–3.4)."""

from __future__ import annotations


def should_generate_padding(*, app_limited: bool, pacing_gain: float) -> bool:
    """Padding when application-limited and BBR is probing (pacing_gain > 1)."""
    return app_limited and pacing_gain > 1.0


def padding_bytes(*, pacing_rate: float, rtt_s: float, app_bytes: float) -> float:
    """Fill to pacing budget so the transport layer is not application-limited."""
    budget = pacing_rate * rtt_s
    return max(0.0, budget - app_bytes)
