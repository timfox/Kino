"""Free-form vs deterministic workflow tool stability (Table V, Sec. IV-H)."""

from __future__ import annotations

import random

from ltx_trainer.trajgenagent.benchmarks import TABLE_V_TOOL_STABILITY
from ltx_trainer.trajgenagent.config import TrajGenAgentConfig
from ltx_trainer.trajgenagent.orchestrator import synthesize_activity_chain_stub
from ltx_trainer.trajgenagent.workflow import run_worker_workflow


def paper_tool_stability(strategy: str) -> dict[str, float]:
    row = next(r for r in TABLE_V_TOOL_STABILITY if r["strategy"] == strategy)
    return {
        "trajectory_success": float(row["trajectory_success"]),
        "visit_success": float(row["visit_success"]),
    }


def simulate_freeform_grounding(
    chain: tuple[str, ...],
    *,
    n_trajectories: int = 200,
    per_visit_fail_p: float = 0.407,
    seed: int = 0,
) -> dict[str, float]:
    """
    Monte Carlo stub for autonomous tool calling fragility (7-activity setting).

    Each visit needs 2 tool calls; any failure breaks downstream state.
    """
    rng = random.Random(seed)
    visit_ok = 0
    visit_total = 0
    traj_ok = 0
    for _ in range(n_trajectories):
        failed = False
        for _visit in chain:
            visit_total += 1
            if failed or rng.random() < per_visit_fail_p:
                failed = True
                continue
            visit_ok += 1
        if not failed:
            traj_ok += 1
    return {
        "trajectory_success": traj_ok / n_trajectories,
        "visit_success": visit_ok / max(1, visit_total),
        "n_trajectories": n_trajectories,
        "chain_length": len(chain),
    }


def tool_stability_comparison(cfg: TrajGenAgentConfig | None = None) -> dict[str, object]:
    cfg = cfg or TrajGenAgentConfig()
    chain = tuple(synthesize_activity_chain_stub())
    workflow = run_worker_workflow(chain, cfg=cfg)
    freeform_live = simulate_freeform_grounding(chain, seed=42)
    paper_free = paper_tool_stability("Free-form tool calling")
    paper_det = paper_tool_stability("Deterministic workflow")
    return {
        "activity_chain_len": len(chain),
        "deterministic_workflow": {
            "live": {
                "trajectory_success": workflow["trajectory_success"],
                "visit_success": workflow["visit_success"],
            },
            "paper": paper_det,
        },
        "freeform_tool_calling": {
            "live": freeform_live,
            "paper": paper_free,
        },
        "workflow_beats_freeform_traj": workflow["trajectory_success"] > freeform_live["trajectory_success"],
    }
