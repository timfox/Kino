"""Sequential vs single-op dispatch measurement (Sec. 7.2, Table 6)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.torch_webgpu.config import PER_DISPATCH_DAWN_US


def sequential_dispatch_total_us(
    n_dispatches: int,
    per_dispatch_us: float,
    *,
    sync_once: bool = True,
) -> float:
    """Wall time for N dispatches with sync only at end (true per-dispatch cost)."""
    sync_us = 450.0 if sync_once else 0.0
    return n_dispatches * per_dispatch_us + sync_us


def single_op_dispatch_us(per_dispatch_us: float, sync_us: float = 450.0) -> float:
    """Naive single-op benchmark: one dispatch + full GPU-CPU sync each iteration."""
    return per_dispatch_us + sync_us


def inflation_factor(per_dispatch_us: float, sync_us: float = 450.0) -> float:
    """Ratio single-op / sequential per-dispatch (paper reports ~20× for Dawn)."""
    seq = per_dispatch_us
    single = single_op_dispatch_us(per_dispatch_us, sync_us)
    return single / seq if seq > 0 else float("inf")


def dispatch_profiler_stub(
    n_dispatches: int = 100,
    *,
    per_dispatch_us: float = PER_DISPATCH_DAWN_US,
    seed: int = 0,
) -> dict[str, Any]:
    """CPU stub mimicking csrc/core/dispatch_profiler.cpp breakdown (Table 20)."""
    rng = np.random.default_rng(seed)
    # Table 20 proportions (submit dominates ~40%)
    components = {
        "encoder_create": 6.4,
        "pass_begin": 3.2,
        "set_pipeline": 1.4,
        "set_bind_group": 1.0,
        "dispatch_call": 0.6,
        "pass_end": 0.7,
        "encoder_finish": 6.1,
        "submit": 12.9,
    }
    noise = {k: v * (1.0 + 0.02 * rng.standard_normal()) for k, v in components.items()}
    per_total = sum(noise.values())
    seq_us = sequential_dispatch_total_us(n_dispatches, per_dispatch_us)
    return {
        "n_dispatches": n_dispatches,
        "per_dispatch_us_measured": per_total,
        "sequential_total_us": seq_us,
        "single_op_us": single_op_dispatch_us(per_dispatch_us),
        "inflation_factor": inflation_factor(per_dispatch_us),
        "timeline_us": noise,
        "submit_fraction": noise["submit"] / per_total,
    }


def dispatch_demo(*, seed: int = 0) -> dict[str, Any]:
    prof = dispatch_profiler_stub(100, seed=seed)
    return {
        "sequential_per_dispatch_us": prof["per_dispatch_us_measured"],
        "single_op_us": prof["single_op_us"],
        "inflation_factor": round(prof["inflation_factor"], 1),
        "submit_dominates": prof["submit_fraction"] > 0.35,
    }
