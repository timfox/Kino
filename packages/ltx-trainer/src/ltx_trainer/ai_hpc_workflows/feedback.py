"""Explicit feedback loops (Tip 9)."""

from __future__ import annotations

from typing import Any


def feedback_loop_spec(
    *,
    max_iterations: int = 5,
    spawn_simulations: bool = True,
    terminate_unproductive: bool = True,
) -> dict[str, Any]:
    return {
        "loop": "simulate → train/refine → infer → validate → decide",
        "max_iterations": max_iterations,
        "controls": {
            "spawn_simulations": spawn_simulations,
            "terminate_unproductive": terminate_unproductive,
            "redirect_compute": True,
        },
        "state_tracked": [
            "model_checkpoint_id",
            "simulation_batch_id",
            "validation_metrics",
            "workflow_generation",
        ],
        "anti_pattern": "manual copy/paste between stages without recorded state",
    }


def iteration_gate(metrics: dict[str, float], *, min_improvement: float = 0.01) -> dict[str, Any]:
    """Simple continue/stop gate for adaptive loop demo."""
    prev = metrics.get("prev_loss")
    curr = metrics.get("curr_loss")
    if prev is None or curr is None:
        return {"action": "continue", "reason": "insufficient history"}
    improved = (prev - curr) / max(abs(prev), 1e-9) >= min_improvement
    return {
        "action": "continue" if improved else "stop",
        "improvement": round((prev - curr) / max(abs(prev), 1e-9), 4),
        "reason": "validation improved" if improved else "diminishing returns",
    }
