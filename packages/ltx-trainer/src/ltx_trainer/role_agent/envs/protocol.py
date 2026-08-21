"""Common agent environment protocol for Role-Agent rollouts."""

from __future__ import annotations

from typing import Protocol

from ltx_trainer.role_agent.types import TaskSpec


class AgentEnv(Protocol):
    task: TaskSpec
    state: str
    done: bool
    success: bool

    def reset(self) -> str: ...

    def step_action(self, action: str) -> tuple[str, float, bool]: ...

    def valid_actions(self) -> list[str]: ...
