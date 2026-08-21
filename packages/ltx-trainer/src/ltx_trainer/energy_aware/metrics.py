"""Facility and AI energy metrics (Sec. 3)."""

from __future__ import annotations


def power_usage_effectiveness(it_power_kw: float, total_facility_power_kw: float) -> float:
    """PUE = total facility power / IT equipment power (≥ 1)."""
    if it_power_kw <= 0:
        raise ValueError("it_power_kw must be positive")
    return float(total_facility_power_kw / it_power_kw)


def water_usage_effectiveness(liters_water: float, it_energy_kwh: float) -> float:
    """WUE in L/kWh."""
    if it_energy_kwh <= 0:
        raise ValueError("it_energy_kwh must be positive")
    return float(liters_water / it_energy_kwh)


def carbon_kg_co2e(power_mw: float, hours: float, kg_co2_per_kwh: float) -> float:
    """Operational carbon from sustained power draw."""
    kwh = power_mw * 1000.0 * hours
    return float(kwh * kg_co2_per_kwh)


def hourly_electricity_cost_usd(power_mw: float, usd_per_kwh: float) -> float:
    """OpEx electricity cost for one hour at sustained MW."""
    return float(power_mw * 1000.0 * usd_per_kwh)


def joules_per_token(total_joules: float, num_tokens: int) -> float:
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    return float(total_joules / num_tokens)


def tokens_per_watt(tokens_per_second: float, watts: float) -> float:
    """Training throughput efficiency (tokens/s/W)."""
    if watts <= 0:
        raise ValueError("watts must be positive")
    return float(tokens_per_second / watts)


def carburacy_proxy(f1_or_quality: float, carbon_kg: float, *, eps: float = 1e-9) -> float:
    """Higher is better: quality per unit carbon (Carburacy family, Sec. 7)."""
    return float(f1_or_quality / max(carbon_kg, eps))
