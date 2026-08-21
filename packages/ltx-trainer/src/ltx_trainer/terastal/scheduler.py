"""Online scheduler — Algorithm 2 (Section IV-C)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReadyLayer:
    request_id: str
    model: str
    layer: int
    arrival_s: float
    virtual_deadline_s: float
    latencies_us: dict[str, float]  # accel_id -> latency
    variant_latencies_us: dict[str, float] | None = None
    variant_allowed: bool = True


@dataclass
class SchedulerState:
    time_s: float = 0.0
    accel_free_at: dict[str, float] = field(default_factory=dict)
    miss_count: int = 0
    completed: int = 0


def finish_time(free_at: float, latency_us: float) -> float:
    return free_at + latency_us / 1e6


def layer_slack(virtual_deadline_s: float, finish_s: float) -> float:
    return virtual_deadline_s - finish_s


def best_case_slack(layer: ReadyLayer, state: SchedulerState) -> float:
    """Eq. 7: max_k slack over accelerators."""
    slacks = []
    for accel, lat_us in layer.latencies_us.items():
        tf = finish_time(state.accel_free_at.get(accel, state.time_s), lat_us)
        slacks.append(layer_slack(layer.virtual_deadline_s, tf))
    if layer.variant_latencies_us and layer.variant_allowed:
        for accel, lat_us in layer.variant_latencies_us.items():
            tf = finish_time(state.accel_free_at.get(accel, state.time_s), lat_us)
            slacks.append(layer_slack(layer.virtual_deadline_s, tf))
    return max(slacks) if slacks else float("-inf")


def schedule_round(
    ready: list[ReadyLayer],
    state: SchedulerState,
) -> tuple[list[dict[str, Any]], SchedulerState]:
    """One invocation of Algorithm 2 when accelerators become idle."""
    assignments: list[dict[str, Any]] = []
    aidle = {a for a, t in state.accel_free_at.items() if t <= state.time_s}
    pending = sorted(ready, key=lambda L: best_case_slack(L, state))

    for layer in list(pending):
        if not aidle:
            break
        # Stage 1a: original layer on feasible idle accel
        cands = []
        for accel in aidle:
            tf = finish_time(state.accel_free_at[accel], layer.latencies_us[accel])
            if tf <= layer.virtual_deadline_s:
                cands.append((tf, accel, False))
        if cands:
            tf, accel, _ = min(cands, key=lambda x: x[0])
            assignments.append(
                {
                    "request": layer.request_id,
                    "layer": layer.layer,
                    "accel": accel,
                    "variant": False,
                    "finish_s": tf,
                }
            )
            state.accel_free_at[accel] = tf
            aidle.remove(accel)
            pending.remove(layer)
            state.completed += 1
            continue
        # Stage 1b: variant if feasible
        if layer.variant_latencies_us and layer.variant_allowed:
            cands = []
            for accel in aidle:
                lat = layer.variant_latencies_us.get(accel)
                if lat is None:
                    continue
                tf = finish_time(state.accel_free_at[accel], lat)
                if tf <= layer.virtual_deadline_s:
                    cands.append((tf, accel, True))
            if cands:
                tf, accel, use_var = min(cands, key=lambda x: x[0])
                assignments.append(
                    {
                        "request": layer.request_id,
                        "layer": layer.layer,
                        "accel": accel,
                        "variant": use_var,
                        "finish_s": tf,
                    }
                )
                state.accel_free_at[accel] = tf
                aidle.remove(accel)
                pending.remove(layer)
                state.completed += 1

    # Stage 2: backfill remaining idle accelerators (slack gain stub)
    for accel in list(aidle):
        if not pending:
            break
        layer = pending[0]
        tf = finish_time(state.accel_free_at[accel], layer.latencies_us[accel])
        assignments.append(
            {
                "request": layer.request_id,
                "layer": layer.layer,
                "accel": accel,
                "variant": False,
                "finish_s": tf,
                "backfill": True,
            }
        )
        state.accel_free_at[accel] = tf
        pending.pop(0)
        state.completed += 1

    return assignments, state


def deadline_miss_rate(
    misses: int,
    total_requests: int,
) -> float:
    if total_requests <= 0:
        return 0.0
    return misses / total_requests


def terastal_vs_baseline_reduction(baseline_miss: float, terastal_miss: float) -> float:
    if baseline_miss <= 0:
        return 0.0
    return (baseline_miss - terastal_miss) / baseline_miss
