"""Routed multi-agent workflow stub: discussion → parallel team execution."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from ltx_trainer.autoscientists.noise_gate import NoiseGate
from ltx_trainer.autoscientists.state import ExperimentRecord, ForumPost, SharedState, TeamState


@dataclass
class WorkflowResult:
    state: SharedState
    trace: list[str] = field(default_factory=list)
    keeps: int = 0
    experiments: int = 0


def _toy_delta(team: str, rng: random.Random) -> tuple[float, float, str]:
    """Return (delta, new_metric, axis) for a synthetic experiment."""
    effects = {
        "architecture": (0.004, "ASPECT_RATIO"),
        "schedule": (0.003, "DEPTH"),
        "throughput": (0.006, "TOTAL_BATCH_SIZE"),
    }
    base_delta, axis = effects.get(team, (0.001, "misc"))
    jitter = rng.uniform(-0.002, 0.008)
    return base_delta + jitter, axis


def run_discussion(state: SharedState, teams: tuple[str, ...]) -> None:
    state.discussion_round += 1
    state.forum.append(
        ForumPost("DISCUSSION-TRIGGER", "analyst0", "Stagnation or cold start — form teams")
    )
    for i, name in enumerate(teams):
        state.teams[name] = TeamState(
            name=name,
            axis=name,
            members=[f"exp{i}", f"analyst{i % 3}"],
            queue=[f"probe_{name}_a", f"probe_{name}_b"],
        )
    state.roster_locked = True
    state.forum.append(ForumPost("ROSTER", "analyst2", f"Teams: {', '.join(teams)}"))


def run_execution_cycle(
    state: SharedState,
    *,
    max_experiments: int = 40,
    seed: int = 0,
    lower_is_better: bool = True,
) -> WorkflowResult:
    """Simulate parallel teams proposing and executing under noise gate."""
    rng = random.Random(seed)
    gate = NoiseGate()
    trace: list[str] = []
    keeps = 0
    metric = state.champion_metric

    if not state.roster_locked:
        run_discussion(state, ("architecture", "schedule", "throughput"))

    for exp_i in range(max_experiments):
        team_name = rng.choice(list(state.teams.keys()))
        team = state.teams[team_name]
        delta_raw, axis = _toy_delta(team_name, rng)
        if lower_is_better:
            delta = -delta_raw
            new_metric = metric + delta
            improved = delta < 0
        else:
            delta = delta_raw
            new_metric = metric + delta
            improved = delta > 0

        accepted = False
        if improved:
            if lower_is_better:
                promote = gate.promote(-delta, confirm_fn=lambda: rng.random() > 0.3)
            else:
                promote = gate.promote(delta, confirm_fn=lambda: rng.random() > 0.3)
            if promote:
                metric = new_metric
                state.champion_metric = metric
                accepted = True
                keeps += 1
                trace.append(f"KEEP:{team_name}:{axis}:{metric:.4f}")

        state.experiment_log.append(
            ExperimentRecord(
                team=team_name,
                axis=axis,
                delta=delta,
                metric=new_metric if improved else metric,
                accepted=accepted,
                proposal=team.queue[0] if team.queue else axis,
            )
        )
        if not accepted and abs(delta_raw) < 0.001:
            team.dead_ends.append(axis)

    state.forum.append(
        ForumPost("RESULT", "system", f"{keeps} KEEPs over {max_experiments} experiments")
    )
    return WorkflowResult(state=state, trace=trace, keeps=keeps, experiments=max_experiments)


def compare_to_autoresearch_baseline(
    *,
    target_metric: float = 0.978,
    autoscientists_experiments: int = 34,
    autoresearch_experiments: int = 65,
    seed: int = 0,
) -> dict[str, Any]:
    """Toy replication of GPT nanochat speedup (parallel teams vs single thread)."""
    _ = seed
    as_hist = [0.998]
    ar_hist = [0.998]
    for i in range(1, autoresearch_experiments + 1):
        # Single-agent: slower early progress, one axis per experiment
        step = 0.00025 if i < 20 else 0.00012
        ar_hist.append(min(ar_hist[-1], ar_hist[-1] - step))
    for i in range(1, autoscientists_experiments + 1):
        # Teams: steeper descent from parallel architecture/schedule/throughput probes
        step = 0.00055 if i < 18 else 0.00015
        as_hist.append(min(as_hist[-1], as_hist[-1] - step))

    def first_hit(hist: list[float], target: float) -> int:
        for i, v in enumerate(hist):
            if v <= target:
                return i
        return len(hist)

    as_n = first_hit(as_hist, target_metric)
    ar_n = first_hit(ar_hist, target_metric)
    return {
        "target_val_bpb": target_metric,
        "autoscientists_experiments_to_target": as_n,
        "autoresearch_experiments_to_target": ar_n,
        "speedup_factor": ar_n / max(1, as_n),
    }
