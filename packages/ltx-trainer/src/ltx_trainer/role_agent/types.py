"""Role-Agent trajectory and batch types (Algorithm 1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ltx_trainer.role_agent.wia import StepRecord


@dataclass
class TaskSpec:
    task_id: str
    domain: str
    prompt: str
    objective: str = ""


@dataclass
class RolloutTrajectory:
    """Single rollout τ = {(s_t, a_t, r_t)} with WIA predictions."""

    task: TaskSpec
    steps: list[StepRecord] = field(default_factory=list)
    predictions: dict[int, dict[int, str]] = field(default_factory=dict)
    success: bool = False
    total_return: float = 0.0
    step_returns: list[float] = field(default_factory=list)
    mixed_advantages: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.steps)

    def states(self) -> list[str]:
        return [s.state for s in self.steps]

    def actions(self) -> list[str]:
        return [s.action for s in self.steps]

    def rewards(self) -> list[float]:
        return [s.reward for s in self.steps]


@dataclass
class RolloutBatch:
    """Batch T = {τ_i} for one training iteration."""

    trajectories: list[RolloutTrajectory] = field(default_factory=list)

    def failed(self) -> list[RolloutTrajectory]:
        return [t for t in self.trajectories if not t.success]

    def succeeded(self) -> list[RolloutTrajectory]:
        return [t for t in self.trajectories if t.success]
