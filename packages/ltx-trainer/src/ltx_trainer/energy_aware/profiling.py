"""Profiling and estimation helpers (Sec. 5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PowerSample:
    watts: float
    source: str  # nvml | rapl | estimate | codecarbon


def estimate_job_energy_joules(power_w_avg: float, duration_s: float) -> float:
    return float(power_w_avg * duration_s)


def estimate_job_energy_kwh(power_w_avg: float, duration_s: float) -> float:
    return estimate_job_energy_joules(power_w_avg, duration_s) / 3.6e6


def estimate_node_power_w(
    cpu_w: float,
    gpu_w: float,
    *,
    memory_w: float = 50.0,
    overhead_fraction: float = 0.12,
) -> float:
    base = cpu_w + gpu_w + memory_w
    return float(base * (1.0 + overhead_fraction))


def gpu_energy_from_utilization(
    tdp_w: float,
    utilization: float,
    duration_s: float,
    *,
    idle_fraction: float = 0.25,
) -> float:
    """
    Rough duty-cycle model when only utilization is known (NVML sampling caveat).

    Paper notes nvidia-smi can undersample; apply margin externally.
    """
    util = max(0.0, min(1.0, utilization))
    avg_w = tdp_w * (idle_fraction + (1.0 - idle_fraction) * util)
    return estimate_job_energy_joules(avg_w, duration_s)


def observer_effect_overhead_fraction(poll_hz: float) -> float:
    """Heuristic: MSR polling >100 Hz adds measurable overhead (Sec. 5.1)."""
    if poll_hz <= 50:
        return 0.0
    return min(0.05, (poll_hz - 50) / 2000.0)


def profiling_report(
    *,
    power_w: float,
    duration_s: float,
    kg_co2_per_kwh: float = 0.35,
    poll_hz: float = 10.0,
) -> dict[str, Any]:
    kwh = estimate_job_energy_kwh(power_w, duration_s)
    return {
        "power_w_avg": power_w,
        "duration_s": duration_s,
        "energy_kwh": round(kwh, 6),
        "energy_j": round(estimate_job_energy_joules(power_w, duration_s), 2),
        "carbon_kg_co2e": round(kwh * kg_co2_per_kwh, 4),
        "observer_overhead_fraction": observer_effect_overhead_fraction(poll_hz),
    }
