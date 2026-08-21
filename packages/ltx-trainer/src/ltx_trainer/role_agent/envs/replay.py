"""JSON replay traces with admissible-command lists (ALFWorld / WebShop / search-QA shape)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ltx_trainer.role_agent.types import TaskSpec

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@dataclass
class ReplayTraceEnv:
    """Text env loaded from fixture JSON (admissible actions per observation)."""

    task: TaskSpec
    state: str = ""
    step: int = 0
    done: bool = False
    success: bool = False
    _graph: dict[str, dict[str, tuple[str, str]]] = field(default_factory=dict)
    _start: str = ""
    _max_steps: int = 50
    backend: str = "replay"

    def reset(self) -> str:
        self.state = self._start
        self.step = 0
        self.done = False
        self.success = False
        return self.state

    def valid_actions(self) -> list[str]:
        return list(self._graph.get(self.state, {}).keys())

    def step_action(self, action: str) -> tuple[str, float, bool]:
        if self.done:
            return self.state, 0.0, True
        self.step += 1
        key = action.strip().lower()
        transitions = self._graph.get(self.state, {})
        if key not in transitions:
            if self.step >= self._max_steps:
                self.done = True
                return f"{self.state}\nEpisode ended: max steps.", 0.0, True
            return f"{self.state}\nNothing happens (invalid: {action}).", 0.0, False
        next_state, note = transitions[key]
        self.state = next_state
        reward = 0.0
        if note == "success":
            self.done = True
            self.success = True
            reward = 1.0
        elif note == "fail":
            self.done = True
            self.success = False
        elif self.step >= self._max_steps:
            self.done = True
        return self.state, reward, self.done


def _load_graph(raw: dict[str, list[dict[str, str]]]) -> dict[str, dict[str, tuple[str, str]]]:
    graph: dict[str, dict[str, tuple[str, str]]] = {}
    for obs, edges in raw.items():
        row: dict[str, tuple[str, str]] = {}
        for edge in edges:
            row[edge["action"].strip().lower()] = (edge["next"], edge.get("note", "ok"))
        graph[obs] = row
    return graph


def replay_env_from_dict(data: dict[str, Any]) -> ReplayTraceEnv:
    task = TaskSpec(
        task_id=str(data["task_id"]),
        domain=str(data["domain"]),
        prompt=str(data["prompt"]),
        objective=str(data.get("objective", "")),
    )
    env = ReplayTraceEnv(
        task=task,
        _graph=_load_graph(data["graph"]),
        _start=str(data["start"]),
        _max_steps=int(data.get("max_steps", 50)),
    )
    return env


def list_replay_fixtures() -> list[str]:
    return sorted(p.stem for p in _FIXTURES_DIR.glob("*.json"))


def make_replay_env(fixture_name: str) -> ReplayTraceEnv:
    stem = fixture_name.removesuffix(".json")
    path = _FIXTURES_DIR / f"{stem}.json"
    if not path.is_file():
        raise KeyError(f"unknown replay fixture: {fixture_name}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return replay_env_from_dict(data)


REPLAY_TASK_POOL: dict[str, ReplayTraceEnv] = {
    name: make_replay_env(name) for name in list_replay_fixtures()
}
