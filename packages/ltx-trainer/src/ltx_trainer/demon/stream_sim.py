"""Multi-tick StreamPipeline simulation (DEMON §3.2, §4.8.2)."""

from __future__ import annotations

from typing import Any, Literal

from ltx_trainer.demon.config import DemonConfig
from ltx_trainer.demon.ring_buffer import (
    Slot,
    admit_slot,
    global_reset_slots,
    heterogeneous_timesteps,
    make_schedule,
)


def _advance_slots(slots: list[Slot | None], *, steps: int) -> tuple[list[Slot | None], int]:
    out: list[Slot | None] = []
    completions = 0
    for s in slots:
        if s is None:
            out.append(None)
            continue
        s.step += 1
        if s.step >= steps:
            completions += 1
            out.append(None)
        else:
            out.append(s)
    return out, completions


def run_stream(
    denoise_schedule: list[float],
    *,
    depth: int = 8,
    steps: int = 8,
    mode: Literal["per_slot", "global_reset"] = "per_slot",
) -> dict[str, Any]:
    """Simulate ring-buffer ticks with per-slot or global-reset denoise handling."""
    slots: list[Slot | None] = [None] * depth
    cache: dict[float, list[float]] = {}
    prev_denoise: float | None = None
    total_completions = 0
    tick_completions: list[int] = []
    heterogeneous_ticks = 0

    for denoise in denoise_schedule:
        if mode == "global_reset" and prev_denoise is not None and denoise != prev_denoise:
            slots = global_reset_slots(slots, depth)
        slots, done = _advance_slots(slots, steps=steps)
        total_completions += done
        tick_completions.append(done)
        admit_slot(slots, denoise=denoise, depth=depth, schedule_cache=cache)
        ts = heterogeneous_timesteps([s for s in slots if s is not None], cache)
        if len(set(ts)) > 1:
            heterogeneous_ticks += 1
        prev_denoise = denoise

    n_ticks = len(denoise_schedule)
    return {
        "mode": mode,
        "n_ticks": n_ticks,
        "completions": total_completions,
        "completion_rate": total_completions / max(n_ticks, 1),
        "heterogeneous_ticks": heterogeneous_ticks,
        "tick_completions": tick_completions,
    }


def linear_denoise_sweep(n_ticks: int, *, lo: float = 0.5, hi: float = 1.0) -> list[float]:
    """1.0→0.5→1.0 style sweep (Table 13 stress test)."""
    half = n_ticks // 2
    down = [hi - (hi - lo) * i / max(half - 1, 1) for i in range(half)]
    up = [lo + (hi - lo) * i / max(n_ticks - half - 1, 1) for i in range(n_ticks - half)]
    return down + up


def ablation_from_simulation(
    *,
    depth: int = 8,
    steps: int = 8,
    sweep_ticks: int = 60,
) -> dict[str, float]:
    """Derive Table 13 completion rates from simulated sweeps, not config constants."""
    schedule = linear_denoise_sweep(sweep_ticks)
    per_slot = run_stream(schedule, depth=depth, steps=steps, mode="per_slot")
    reset = run_stream(schedule, depth=depth, steps=steps, mode="global_reset")
    warmup = depth + steps - 1
    steady = max(len(schedule) - warmup, 1)
    per_slot_steady = sum(per_slot["tick_completions"][warmup:]) / steady
    reset_steady = sum(reset["tick_completions"][warmup:]) / steady
    return {
        "per_slot_sweep": per_slot_steady,
        "global_reset_sweep": reset_steady,
    }


def stream_smoke(cfg: DemonConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DemonConfig()
    depth = cfg.max_depth
    steps = cfg.denoising_steps
    warmup = run_stream([1.0] * depth, depth=depth, steps=steps, mode="per_slot")
    switch = run_stream([1.0] * depth + [0.5] * (steps + 2), depth=depth, steps=steps, mode="per_slot")
    ablation = ablation_from_simulation(depth=depth, steps=steps)
    return {
        "warmup_completions": warmup["completions"],
        "post_switch_heterogeneous": switch["heterogeneous_ticks"] > 0,
        "ablation": ablation,
        "per_slot_beats_reset": ablation["per_slot_sweep"] > ablation["global_reset_sweep"],
    }
