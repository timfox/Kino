"""BES search types: trajectories, goals, callbacks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

Step = str
Trajectory = tuple[Step, ...]


@dataclass(frozen=True)
class Node:
    """Partial trajectory in the forward pool."""

    steps: Trajectory

    def __len__(self) -> int:
        return len(self.steps)

    def is_terminal(self, terminal_check: Callable[[Trajectory], bool]) -> bool:
        return terminal_check(self.steps)


@dataclass
class Goal:
    """Sub-goal in the backward tree."""

    goal_id: str
    description: str
    parent_id: str | None = None
    children: list[str] = field(default_factory=list)

    def is_leaf(self) -> bool:
        return len(self.children) == 0


@dataclass
class GoalTree:
    """Backward decomposition tree rooted at groot."""

    goals: dict[str, Goal] = field(default_factory=dict)
    root_id: str = "groot"

    def leaves(self) -> list[Goal]:
        return [g for g in self.goals.values() if g.is_leaf()]

    def children_of(self, goal_id: str) -> list[Goal]:
        g = self.goals[goal_id]
        return [self.goals[cid] for cid in g.children if cid in self.goals]


VerifierFn = Callable[[Trajectory], float]
PolicyFn = Callable[[Trajectory, int], list[Step]]
DecomposeFn = Callable[[str, GoalTree], list[tuple[str, str, VerifierFn]]]
