"""Scheduling overhead model (§3.2, Eq. 1–4)."""

from __future__ import annotations

from typing import Any


def ideal_time(
    batch_size: int,
    tin: float,
    tk: float,
    tout: float,
) -> float:
    """T_ideal = b·t_in + t_k + t_out."""
    return batch_size * tin + tk + tout


def intra_batch_overhead(
    batch_size: int,
    tin_in: float,
    tin_k: float,
    delta_tk: float,
    tk_out: float,
) -> float:
    """t_intra = (b-1)t_in-in + t_in-k + Δt_k + t_k-out."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    return (batch_size - 1) * tin_in + tin_k + delta_tk + tk_out


def inter_batch_overhead(t_start_batch2: float, t_end_batch1: float) -> float:
    """t_inter = t_start_batch2 - t_end_batch1."""
    return t_start_batch2 - t_end_batch1


def measured_time(
    t_launch: float,
    t_sync: float,
    t_host: float,
) -> float:
    """T_measured = t_launch + t_sync + t_host."""
    return t_launch + t_sync + t_host


def schedule_fraction(t_schedule: float, t_measured: float) -> float:
    """Fraction = t_schedule / T_measured (Fig. 6)."""
    if t_measured <= 0:
        raise ValueError("t_measured must be positive")
    return t_schedule / t_measured


def decompose_measured(
    *,
    batch_size: int,
    tin: float,
    tk: float,
    tout: float,
    tin_in: float,
    tin_k: float,
    delta_tk: float,
    tk_out: float,
    t_inter: float,
) -> dict[str, float]:
    """Return T_ideal, t_intra, t_inter, t_schedule, T_measured."""
    t_ideal = ideal_time(batch_size, tin, tk, tout)
    t_intra = intra_batch_overhead(batch_size, tin_in, tin_k, delta_tk, tk_out)
    t_schedule = t_intra + t_inter
    t_meas = t_ideal + t_schedule
    return {
        "T_ideal": t_ideal,
        "t_intra": t_intra,
        "t_inter": t_inter,
        "t_schedule": t_schedule,
        "T_measured": t_meas,
        "schedule_fraction": schedule_fraction(t_schedule, t_meas),
    }


def overhead_model_card() -> dict[str, Any]:
    return {
        "T_ideal": "b·t_in + t_k + t_out",
        "t_intra": "(b-1)t_in-in + t_in-k + Δt_k + t_k-out",
        "t_inter": "t_start_batch2 - t_end_batch1",
        "T_measured": "T_ideal + t_schedule = t_launch + t_sync + t_host",
        "insight": "U-shaped schedule fraction vs batch size; inter-batch dominates at large b",
    }
