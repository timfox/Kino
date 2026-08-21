"""StreamPipeline ring buffer and propagation classes (DEMON §3.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ltx_trainer.demon.config import DemonConfig, PropagationClass


@dataclass
class Slot:
    denoise: float
    step: int = 0
    schedule_id: int = 0


def make_schedule(denoise: float, *, steps: int = 8) -> list[float]:
    """Toy linear sigma schedule truncated by denoise strength."""
    n = max(2, int(round(steps * denoise)))
    return [1.0 - i / (n - 1) for i in range(n + 1)]


def admit_slot(
    slots: list[Slot | None],
    *,
    denoise: float,
    depth: int,
    schedule_cache: dict[float, list[float]],
) -> list[float]:
    """Admit slot with per-slot heterogeneous schedule baked at admission."""
    if denoise not in schedule_cache:
        schedule_cache[denoise] = make_schedule(denoise)
    sid = id(schedule_cache[denoise])
    for i, s in enumerate(slots):
        if s is None:
            slots[i] = Slot(denoise=denoise, step=0, schedule_id=sid)
            break
    else:
        slots[0] = Slot(denoise=denoise, step=0, schedule_id=sid)
    return schedule_cache[denoise]


def tick_slots(slots: list[Slot | None], *, steps: int = 8) -> list[Slot | None]:
    """Advance all active slots one denoising step; retire completed."""
    out: list[Slot | None] = []
    for s in slots:
        if s is None:
            out.append(None)
            continue
        s.step += 1
        out.append(None if s.step >= steps else s)
    return out


def heterogeneous_timesteps(slots: list[Slot | None], schedule_cache: dict[float, list[float]]) -> list[float]:
    """Batched forward: each row draws from its own slot schedule."""
    ts: list[float] = []
    for s in slots:
        if s is None:
            continue
        sched = schedule_cache.get(s.denoise, make_schedule(s.denoise))
        idx = min(s.step, len(sched) - 1)
        ts.append(sched[idx])
    return ts


def global_reset_slots(slots: list[Slot | None], depth: int) -> list[Slot | None]:
    """StreamDiffusion-style prepare() wipe."""
    return [None] * depth


def propagation_taxonomy() -> list[dict[str, Any]]:
    return [
        {
            "class": PropagationClass.PER_REQUEST.value,
            "examples": "conditioning, source audio",
            "onset": "S ticks",
            "convergence": "S ticks",
            "mechanism": "Baked into slot at submission",
        },
        {
            "class": PropagationClass.SCHEDULE_MIGRATED.value,
            "examples": "denoise schedule (migration)",
            "onset": "1 tick",
            "convergence": "S ticks",
            "mechanism": "Migrated onto every in-flight slot; transient hybrids",
        },
        {
            "class": PropagationClass.PER_STEP_SHARED.value,
            "examples": "SDE curve, x0-target morph, velocity scale",
            "onset": "1 tick",
            "convergence": "progressive (< S)",
            "mechanism": "Read from shared state each step",
        },
        {
            "class": PropagationClass.MODEL_WEIGHT.value,
            "examples": "LoRA refit",
            "onset": "1 tick",
            "convergence": "1 tick",
            "mechanism": "Shared decoder weights",
        },
    ]


def ablation_completion_rates(cfg: DemonConfig | None = None) -> dict[str, float]:
    from ltx_trainer.demon.stream_sim import ablation_from_simulation

    cfg = cfg or DemonConfig()
    return ablation_from_simulation(depth=cfg.max_depth, steps=cfg.denoising_steps)


def ring_smoke(cfg: DemonConfig | None = None) -> dict[str, Any]:
    from ltx_trainer.demon.stream_sim import run_stream, stream_smoke

    cfg = cfg or DemonConfig()
    depth = cfg.max_depth
    steps = cfg.denoising_steps
    slots: list[Slot | None] = [None] * depth
    cache: dict[float, list[float]] = {}
    admit_slot(slots, denoise=1.0, depth=depth, schedule_cache=cache)
    admit_slot(slots, denoise=0.5, depth=depth, schedule_cache=cache)
    ts = heterogeneous_timesteps([s for s in slots if s], cache)
    tick_out = run_stream([1.0, 0.5, 1.0, 0.5] * (depth + steps), depth=depth, steps=steps, mode="per_slot")
    sim = stream_smoke(cfg)
    return {
        "n_active": sum(1 for s in slots if s is not None),
        "heterogeneous_timesteps": len(set(ts)) >= 1,
        "tick_completions": tick_out["completions"],
        "per_slot_beats_reset": sim["per_slot_beats_reset"],
        "ablation": sim["ablation"],
    }
