"""Configuration for facility / job energy models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FacilityConfig:
    """Data-center or HPC site parameters (Sec. 3)."""

    name: str = "generic"
    power_mw: float = 1.0
    pue: float = 1.2
    kg_co2_per_kwh: float = 0.35
    usd_per_kwh: float = 0.15
    wue_liters_per_kwh: float = 1.8
    renewable_fraction: float = 0.0


@dataclass
class JobEnergyProfile:
    """Per-job telemetry summary for scheduling (Sec. 5–6)."""

    job_id: str
    power_w_avg: float
    duration_s: float
    flops_estimate: float = 0.0
    memory_bound: bool = False
    flexible_start: bool = True
    deadline_hours: float | None = None


@dataclass
class EnergyAwareConfig:
    paper_arxiv: str = "2605.24569"
    paper_title: str = "Energy-Aware Computing in the Year 2026"
    authors: str = "Tchakoute, Tadonki (Mines Paris - PSL)"
    liquid_cooling_rack_kw_threshold: float = 35.0
    default_gpu_tdp_w: float = 700.0
    keywords: tuple[str, ...] = field(
        default_factory=lambda: (
            "energy efficiency",
            "carbon footprint",
            "power capping",
            "green AI",
            "cloud-edge-HPC continuum",
        )
    )
