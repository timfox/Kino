"""Nyquist velocity and interlaced cross-validation helpers."""

from __future__ import annotations


def nyquist_velocity(*, delta_x: float, delta_t: float) -> float:
    """v_Nyquist = Δx / Δt (Eq. 1 in the paper)."""
    if delta_t <= 0:
        raise ValueError("delta_t must be positive")
    return delta_x / delta_t


def evenly_spaced_projection_angles(
    num_projections: int,
    *,
    span_deg: float = 180.0,
) -> list[float]:
    """Evenly spaced angles in [0, span) as used for sparse bootstrap Pi."""
    if num_projections < 2:
        raise ValueError("need at least 2 projections for shared 3D information")
    step = span_deg / num_projections
    return [i * step for i in range(num_projections)]


def interlaced_time_indices(num_frames: int) -> tuple[list[int], list[int]]:
    """Even / odd frame indices for interlaced CV (Eq. 5); skips last if odd count."""
    if num_frames < 2:
        raise ValueError("need at least 2 frames")
    n = num_frames if num_frames % 2 == 0 else num_frames - 1
    even = list(range(0, n, 2))
    odd = list(range(1, n, 2))
    return even, odd
