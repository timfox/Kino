"""Stencil radius, ghost depth, throughput metrics (Sections 3.1, 6)."""

from __future__ import annotations

from ltx_trainer.fdtd_cpml_multigpu.constants import STENCIL_RADIUS


def ghost_depth(comm_interval: int, *, radius: int = STENCIL_RADIUS) -> int:
    """Enlarged ghost depth g = 2 r s (Section 3.1)."""
    return 2 * radius * comm_interval


def throughput_mpoints_per_sec(
    nx: int,
    ny: int,
    nz: int,
    n_steps: int,
    runtime_s: float,
) -> float:
    if runtime_s <= 0:
        return 0.0
    return nx * ny * nz * n_steps / (1e6 * runtime_s)


def strong_scaling_efficiency(t1: float, tp: float, p: int) -> float:
    if tp <= 0 or p <= 0:
        return 0.0
    return 100.0 * t1 / (p * tp)


def cpml_overhead_pct(t_no_cpml: float, t_cpml: float) -> float:
    if t_no_cpml <= 0:
        return 0.0
    return 100.0 * (t_cpml - t_no_cpml) / t_no_cpml


def enlarged_ghost_speedup(t_s1: float, t_s: float) -> float:
    if t_s <= 0:
        return 0.0
    return t_s1 / t_s
