"""Energy-aware scheduling heuristics (Sec. 6.3–6.5)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CarbonWindow:
    """Hour index 0–23 with grid carbon intensity (kg CO2e/kWh)."""

    hour: int
    kg_co2_per_kwh: float


def carbon_aware_start_hour(
    job_duration_h: float,
    windows: list[CarbonWindow],
    *,
    deadline_hour: int | None = None,
) -> int:
    """
    Pick start hour minimizing carbon for flexible workload (temporal shifting).

    ``deadline_hour``: latest hour by which job must complete (exclusive wrap).
    """
    if not windows:
        return 0
    dur = max(1, int(job_duration_h + 0.999))
    best_h, best_c = 0, float("inf")
    hours = sorted({w.hour % 24 for w in windows})
    intensity = {w.hour % 24: w.kg_co2_per_kwh for w in windows}
    for start in hours:
        if deadline_hour is not None and (start + dur) % 24 > deadline_hour % 24 and start > deadline_hour % 24:
            continue
        carbon = sum(intensity.get((start + i) % 24, intensity.get(start, 0.4)) for i in range(dur))
        if carbon < best_c:
            best_c, best_h = carbon, start
    return best_h


def first_fit_bin_pack(item_sizes: list[float], bin_capacity: float) -> tuple[list[list[int]], int]:
    """Green VM consolidation: first-fit decreasing bin packing (indices into item_sizes)."""
    order = sorted(range(len(item_sizes)), key=lambda i: item_sizes[i], reverse=True)
    bins: list[list[int]] = []
    remaining: list[float] = []
    for idx in order:
        size = item_sizes[idx]
        placed = False
        for b, rem in enumerate(remaining):
            if size <= rem:
                bins[b].append(idx)
                remaining[b] -= size
                placed = True
                break
        if not placed:
            bins.append([idx])
            remaining.append(bin_capacity - size)
    return bins, len(bins)


def recommend_dvfs_scale(
    *,
    memory_bound: bool,
    slack_fraction: float,
) -> float:
    """
    Suggest frequency scale ∈ (0,1].

    Memory-bound: lowering f reduces power with little time gain (Sec. 6.5).
    """
    if memory_bound:
        return max(0.6, 1.0 - 0.5 * slack_fraction)
    # CPU-bound race-to-halt prefers high f if slack is low
    if slack_fraction < 0.1:
        return 1.0
    return max(0.75, 1.0 - 0.3 * slack_fraction)


def power_cap_allocation(
    total_cap_w: float,
    demands_w: list[float],
) -> list[float]:
    """Proportional fair split under a SLURM-style node power cap."""
    if not demands_w:
        return []
    total = sum(demands_w)
    if total <= total_cap_w:
        return list(demands_w)
    scale = total_cap_w / total
    return [d * scale for d in demands_w]
