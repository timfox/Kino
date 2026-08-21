"""Single-shot and Iterative workflow phase graphs (§IV–V)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.deopt_reopt.constants import (
    ITERATIVE_GENERATIONS,
    ITERATIVE_PG_PER_GEN,
    SINGLE_SHOT_LLM_CALLS,
)


def single_shot_phases(mode: str) -> list[dict[str, str]]:
    """Return ordered LLM phases for Direct, Deopt-Reopt, or Direct-3."""
    if mode == "direct":
        return [
            {"phase": "translation", "role": "translate C++ → CUDA"},
            {"phase": "reoptimization", "role": "GPU-target reoptimization"},
        ]
    if mode == "deopt_reopt":
        return [
            {"phase": "deoptimization", "role": "remove CPU-specific optimizations"},
            {"phase": "translation", "role": "translate simplified C++ → CUDA"},
            {"phase": "reoptimization", "role": "GPU-target reoptimization"},
        ]
    if mode == "direct_3":
        return [
            {"phase": "translation", "role": "translate C++ → CUDA"},
            {"phase": "reoptimization_v1", "role": "first reoptimization pass"},
            {"phase": "reoptimization_v2", "role": "second reoptimization pass"},
        ]
    raise ValueError(f"unknown mode: {mode}")


def iterative_phases(mode: str) -> list[str]:
    """Stage axis for Figure 3: IN, DO1–DO3, CO, OP1–OP3."""
    stages = ["IN"]
    if mode == "deopt_reopt":
        stages.extend(f"DO{i}" for i in range(1, ITERATIVE_GENERATIONS + 1))
    stages.append("CO")
    stages.extend(f"OP{i}" for i in range(1, ITERATIVE_GENERATIONS + 1))
    return stages


def workflow_card() -> dict[str, Any]:
    return {
        "single_shot": {
            "modes": ("direct", "deopt_reopt", "direct_3"),
            "llm_calls": SINGLE_SHOT_LLM_CALLS,
            "trials_per_kernel_model": 50,
            "total_trials": 3600,
            "selection": "one pass; last passing stage adopted",
        },
        "iterative": {
            "modes": ("direct", "deopt_reopt"),
            "generations": ITERATIVE_GENERATIONS,
            "pg_per_generation": ITERATIVE_PG_PER_GEN,
            "repair_per_pg": 1,
            "pm_role": "project manager proposes per-PG strategy",
            "deopt_selection": "smallest LOC among valid candidates < input LOC",
            "reopt_selection": "smallest runtime among valid candidates",
            "total_trials": 2400,
        },
    }


def deoptimization_contract() -> dict[str, Any]:
    """Constraints preserved through deoptimization (§IV-A)."""
    return {
        "preserve": ("function signature", "problem sizes", "numerical results within tolerance"),
        "remove": ("NEON SIMD expansion", "blocking", "CPU-specific data layouts"),
        "hardware_info": "given to translation/reopt only, not deoptimization",
        "validation": "same driver and per-kernel tolerance as input C++",
    }
