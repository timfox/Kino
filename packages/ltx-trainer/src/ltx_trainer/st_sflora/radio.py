"""Wireless delay, energy, and mobility-aware client selection (Sec. III–IV)."""

from __future__ import annotations

import math

LN2 = math.log(2.0)


def noise_psd_watts_per_hz(noise_dbm_hz: float) -> float:
    return 10.0 ** ((noise_dbm_hz - 30.0) / 10.0)


def uplink_rate_bps(
    bandwidth_hz: float,
    power_w: float,
    channel_gain: float,
    noise_psd: float,
) -> float:
    """R^UL_m (Eq. 3)."""
    if bandwidth_hz <= 0:
        return 0.0
    snr = power_w * channel_gain / (noise_psd * bandwidth_hz)
    return bandwidth_hz * math.log2(1.0 + max(snr, 0.0))


def uplink_latency_s(data_bits: float, rate_bps: float) -> float:
    if rate_bps <= 0:
        return float("inf")
    return data_bits / rate_bps


def uplink_energy_j(power_w: float, latency_s: float) -> float:
    return power_w * latency_s


def standing_time_s(
    coverage_radius_m: float,
    distance_m: float,
    velocity_mps: float,
    max_latency_s: float,
) -> float:
    """T^standing (Eq. 7)."""
    if velocity_mps <= 0:
        return max_latency_s
    travel = max(0.0, coverage_radius_m - distance_m) / velocity_mps
    return min(travel, max_latency_s)


def client_selected(
    standing_s: float,
    holding_s: float,
) -> bool:
    """θ_m = 1 iff holding ≤ standing (Eq. 9)."""
    return holding_s <= standing_s


def min_power_for_latency(
    data_bits: float,
    bandwidth_hz: float,
    channel_gain: float,
    noise_psd: float,
    max_uplink_latency_s: float,
) -> float:
    """p^min from Eq. 27 inverse."""
    if max_uplink_latency_s <= 0 or bandwidth_hz <= 0:
        return float("inf")
    r_req = data_bits / max_uplink_latency_s
    phi = channel_gain / (noise_psd * bandwidth_hz)
    # R = W log2(1 + phi p)  =>  p = (2^{R/W} - 1) / phi
    if phi <= 0:
        return float("inf")
    return (2.0 ** (r_req / bandwidth_hz) - 1.0) / phi
