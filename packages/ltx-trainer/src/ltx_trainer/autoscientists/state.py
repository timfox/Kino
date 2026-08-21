"""Shared experimental state S: champion, log L, forum F, team queues."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExperimentRecord:
    team: str
    axis: str
    delta: float
    metric: float
    accepted: bool
    proposal: str = ""


@dataclass
class ForumPost:
    kind: str
    author: str
    body: str


@dataclass
class TeamState:
    name: str
    axis: str
    queue: list[str] = field(default_factory=list)
    dead_ends: list[str] = field(default_factory=list)
    members: list[str] = field(default_factory=list)


@dataclass
class SharedState:
    task: str
    champion_metric: float
    champion_program: dict[str, Any] = field(default_factory=dict)
    experiment_log: list[ExperimentRecord] = field(default_factory=list)
    forum: list[ForumPost] = field(default_factory=list)
    teams: dict[str, TeamState] = field(default_factory=dict)
    roster_locked: bool = False
    discussion_round: int = 0

    def best_metric(self) -> float:
        if not self.experiment_log:
            return self.champion_metric
        accepted = [e for e in self.experiment_log if e.accepted]
        if not accepted:
            return self.champion_metric
        return min(e.metric for e in accepted) if self._lower_is_better() else max(
            e.metric for e in accepted
        )

    def _lower_is_better(self) -> bool:
        return "bpb" in self.task.lower() or "loss" in self.task.lower() or "mae" in self.task.lower()
