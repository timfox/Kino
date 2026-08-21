"""BES high-level API, toy demos, and legacy helpers."""

from __future__ import annotations

import re
from typing import Any

from ltx_trainer.bes.backward import score_node
from ltx_trainer.bes.config import BESConfig
from ltx_trainer.bes.search import BESSearchResult, bidirectional_evolutionary_search
from ltx_trainer.bes.theory import theory_card
from ltx_trainer.bes.types import Goal, GoalTree, Node, Trajectory, VerifierFn


def bes_forward_score(partial_len: int, total_len: int) -> float:
    """Legacy progress proxy (kept for stub compatibility)."""
    return partial_len / max(total_len, 1)


def bes_backward_subgoals(n: int) -> list[str]:
    """Legacy placeholder sub-goal ids."""
    return [f"subgoal_{i}" for i in range(n)]


def _toy_arithmetic_demo(seed: int = 42) -> BESSearchResult:
    """
    MuSiQue-style case study: decompose (4+6)×3/2 − 5 with embedding-free keyword verifiers.
    """
    cfg = BESConfig(budget_calls=40, decompose_interval=8, tau_start=1.5, tau_end=0.5)

    plan = [
        "identify artist James Blunt recorded Back to Bedlam",
        "note Custard Records as original label",
        "answer: Custard Records",
    ]

    def policy(prefix: Trajectory, k: int) -> list[str]:
        idx = len(prefix)
        out: list[str] = []
        for i in range(k):
            if idx + i < len(plan):
                out.append(plan[idx + i])
            else:
                out.append("refine reasoning")
        return out

    def terminal_check(traj: Trajectory) -> bool:
        text = " ".join(traj).lower()
        return "answer:" in text or "custard" in text

    def root_verifier(traj: Trajectory) -> float:
        text = " ".join(traj).lower()
        if "custard" in text and "answer" in text:
            return 1.0
        if "custard" in text:
            return 0.6
        if "james blunt" in text:
            return 0.35
        return 0.1

    def v_artist(traj: Trajectory) -> float:
        return 1.0 if "james blunt" in " ".join(traj).lower() else 0.0

    def v_label(traj: Trajectory) -> float:
        return 1.0 if "custard" in " ".join(traj).lower() else 0.0

    tree = GoalTree(
        goals={
            "groot": Goal("groot", "answer label question", children=["g1", "g2"]),
            "g1": Goal("g1", "which artist recorded Back to Bedlam"),
            "g2": Goal("g2", "record label of artist"),
        }
    )
    verifiers: dict[str, VerifierFn] = {
        "groot": root_verifier,
        "g1": v_artist,
        "g2": v_label,
    }

    def decompose_fn(goal: Goal, _pool: list[Node]) -> list[tuple[str, str, VerifierFn]]:
        if goal.goal_id == "g2":
            return []
        return [
            ("g1", "find artist", v_artist),
            ("g2", "find label", v_label),
        ]

    return bidirectional_evolutionary_search(
        policy=policy,
        terminal_verifier=root_verifier,
        terminal_check=terminal_check,
        initial_tree=tree,
        initial_verifiers=verifiers,
        decompose_fn=decompose_fn,
        cfg=cfg,
        seed=seed,
    )


def _toy_kk_step_demo() -> dict[str, Any]:
    """Demonstrate evolution operators on synthetic knight/knave steps."""
    from ltx_trainer.bes.operators import combine_trajectories, crossover_trajectories

    a = ("assume Alice knight", "contradiction for Bob")
    b = ("assume Alice knight", "Bob is knave", "consistent")
    combined = combine_trajectories(a, b)
    crossed = crossover_trajectories(a, b, __import__("random").Random(0))
    return {
        "path_a": a,
        "path_b": b,
        "combination": combined,
        "crossover": crossed,
    }


def run_toy_search(seed: int = 42) -> dict[str, Any]:
    """End-to-end CPU demo without LLM API."""
    result = _toy_arithmetic_demo(seed=seed)
    ops = _toy_kk_step_demo()
    return {
        "search": {
            "found_success": result.found_success,
            "best_score": round(result.best_score, 4),
            "policy_calls": result.policy_calls,
            "pool_size": result.pool_size,
            "trace": result.trace,
        },
        "operators": ops,
        "theory": theory_card(),
    }


def post_training_sample_plan(problems_per_epoch: int = 5000) -> dict[str, Any]:
    """How BES replaces i.i.d. rollouts in GRPO / MaxRL / Tree-GRPO."""
    return {
        "mode": "replace_rollout_stage",
        "trajectories_per_problem": 8,
        "search_budget_calls": 200,
        "padding": "single-rollout pad to group_size when <8 terminals",
        "compatible_trainers": ["GRPO", "MaxRL", "Tree-GRPO"],
    }


def score_trajectory_against_tree(
    trajectory: Trajectory,
    tree: GoalTree,
    verifiers: dict[str, VerifierFn],
    *,
    alpha: float = 0.3,
) -> float:
    return score_node(trajectory, tree.root_id, tree, verifiers, alpha=alpha)


def parse_final_answer_kk(text: str) -> dict[str, int] | None:
    """Parse K&K JSON final answer from trajectory text."""
    m = re.search(r"\{[^{}]+\}", text)
    if not m:
        return None
    try:
        import json

        raw = json.loads(m.group(0))
        return {str(k): int(v) for k, v in raw.items()}
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
