"""Toy rollouts for CPU smoke demos."""

from __future__ import annotations

from typing import Any

from ltx_trainer.role_agent.aiw import (
    FailureMemory,
    FailureReflection,
    classify_failure_heuristic,
    curriculum_resample_weights,
    parse_reflection,
)
from ltx_trainer.role_agent.gigpo import mixed_advantages
from ltx_trainer.role_agent.wia import StepRecord, compute_step_returns


def demo_wia_toy_rollout() -> dict[str, Any]:
    """ALFWorld-style mini trajectory with state predictions."""
    traj = [
        StepRecord("You are in the kitchen. You see a apple.", "go to countertop 1", 0.0),
        StepRecord("On countertop 1 you see a knife and apple.", "take apple 1", 0.0),
        StepRecord("You pick up apple 1.", "go to fridge 1", 0.0),
        StepRecord("The fridge 1 is closed.", "open fridge 1", 0.0),
        StepRecord("You open fridge 1. Task success.", "done", 1.0),
    ]
    predictions = {
        0: {1: "On countertop 1 you see a knife and apple.", 2: "You pick up apple 1."},
        2: {1: "The fridge 1 is closed.", 2: "You open fridge 1."},
    }
    returns = compute_step_returns(traj, predictions, horizon=2, gamma=0.99)
    states = [s.state for s in traj]
    actions = [s.action for s in traj]
    rollout_total = sum(s.reward for s in traj)
    adv = mixed_advantages(
        returns,
        states,
        actions,
        rollout_total,
        [rollout_total, rollout_total * 0.2],
        similarity_threshold=0.9,
        alpha=1.0,
    )
    return {
        "step_returns": returns,
        "final_modulated_return": returns[-1],
        "mixed_advantages_last": adv[-1],
        "predictive_improves_success_step": returns[-1] > returns[1],
    }


def demo_aiw_failure_curriculum() -> dict[str, Any]:
    reflection_text = """<reflection>
DOMINANT_TYPE: wrong_receptacle
CORE_LESSON: Verify container type before placing object
RETRIEVAL_QUERY: put apple fridge wrong cabinet
</reflection>"""
    parsed = parse_reflection(reflection_text, task_id="alf_042", domain="alfworld")
    assert parsed is not None
    memory = FailureMemory()
    memory.add(parsed)
    memory.add(
        FailureReflection(
            dominant_type="navigation_loop",
            core_lesson="Avoid revisiting same location without progress",
            retrieval_query="kitchen loop go to",
            task_id="alf_017",
            domain="alfworld",
        )
    )
    heuristic = classify_failure_heuristic(
        "go to shelf 1\ngo to shelf 1\ngo to shelf 1\ninvalid action format",
        "alfworld",
    )
    weights = curriculum_resample_weights(
        ["alf_001", "alf_042", "alf_099"],
        memory,
        boost=2.0,
    )
    retrieved = memory.retrieve("wrong receptacle apple fridge", domain="alfworld")
    return {
        "parsed_mode": parsed.dominant_type,
        "heuristic_mode": heuristic,
        "unique_alfworld_modes": len(memory.unique_modes("alfworld")),
        "boosted_task": max(weights, key=weights.get),
        "retrieved_task_ids": [r.task_id for r in retrieved],
        "ok": weights["alf_042"] == 2.0 and len(retrieved) >= 1,
    }


def demo_dual_role_loop() -> dict[str, Any]:
    wia = demo_wia_toy_rollout()
    aiw = demo_aiw_failure_curriculum()
    return {"wia": wia, "aiw": aiw, "ok": wia["predictive_improves_success_step"] and aiw["ok"]}
