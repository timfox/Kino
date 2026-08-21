"""Toy tabular policy + LLM policy interface for Role-Agent rollouts."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Protocol

from ltx_trainer.role_agent.action_parse import build_agent_prompt, parse_action_from_response


class Policy(Protocol):
    def choose_action(self, state: str, *, valid_actions: list[str] | None = None) -> str: ...

    def log_prob(self, state: str, action: str) -> float: ...

    def update(self, state: str, action: str, advantage: float, *, lr: float) -> None: ...


@dataclass
class ToyLogPolicy:
    """State-hash policy with softmax logits over seen actions (CPU GRPO target)."""

    temperature: float = 1.0
    _logits: dict[str, dict[str, float]] = field(default_factory=dict)
    _rng: random.Random = field(default_factory=random.Random)

    def seed(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def _ensure(self, state: str, actions: list[str]) -> None:
        if state not in self._logits:
            self._logits[state] = {a: 0.0 for a in actions}
        for a in actions:
            self._logits[state].setdefault(a, 0.0)

    def valid_actions_for(self, env_graph: dict[str, dict[str, tuple[str, str]]], state: str) -> list[str]:
        return list(env_graph.get(state, {}).keys())

    def choose_action(self, state: str, *, valid_actions: list[str] | None = None) -> str:
        actions = valid_actions or []
        if not actions:
            return "noop"
        self._ensure(state, actions)
        logits = [self._logits[state][a] / max(self.temperature, 1e-6) for a in actions]
        m = max(logits)
        exps = [math.exp(x - m) for x in logits]
        z = sum(exps)
        probs = [e / z for e in exps]
        r = self._rng.random()
        acc = 0.0
        for action, p in zip(actions, probs, strict=True):
            acc += p
            if r <= acc:
                return action
        return actions[-1]

    def log_prob(self, state: str, action: str) -> float:
        row = self._logits.get(state, {})
        if action not in row:
            return -10.0
        logits = list(row.values())
        actions = list(row.keys())
        m = max(logits)
        exps = [math.exp((row[a] - m) / max(self.temperature, 1e-6)) for a in actions]
        z = sum(exps)
        return math.log(exps[actions.index(action)] / z + 1e-12)

    def update(self, state: str, action: str, advantage: float, *, lr: float) -> None:
        if state not in self._logits or action not in self._logits[state]:
            return
        self._logits[state][action] += lr * advantage


@dataclass
class ScriptedPolicy:
    """Deterministic oracle-ish policy for smoke tests."""

    plan: dict[str, str]

    def choose_action(self, state: str, *, valid_actions: list[str] | None = None) -> str:
        if state in self.plan:
            return self.plan[state]
        if valid_actions:
            return valid_actions[0]
        return "noop"

    def log_prob(self, state: str, action: str) -> float:
        chosen = self.choose_action(state, valid_actions=[action] if action else None)
        return 0.0 if chosen == action else -5.0

    def update(self, state: str, action: str, advantage: float, *, lr: float) -> None:
        return None


@dataclass
class LLMActionPolicy:
    """vLLM-backed action policy for toy text envs (eval / rollout demo; no local weight update)."""

    client: object
    task_prompt: str = ""
    domain: str = "alfworld"
    _last_log_prob: float = -1.0
    _cache_key: tuple[str, str] | None = None

    def with_task(self, *, task_prompt: str, domain: str) -> LLMActionPolicy:
        return LLMActionPolicy(client=self.client, task_prompt=task_prompt, domain=domain)

    def choose_action(self, state: str, *, valid_actions: list[str] | None = None) -> str:
        actions = valid_actions or []
        if not actions:
            return "noop"
        prompt = build_agent_prompt(
            domain=self.domain,
            task=self.task_prompt or "Complete the task.",
            state=state,
            valid_actions=actions,
        )
        try:
            text = self.client.complete(prompt, temperature=self.client.config.rollout_temperature, max_tokens=128)
            action = parse_action_from_response(text, actions, domain=self.domain)
        except Exception:
            action = actions[0]
        self._cache_key = (state, action)
        self._last_log_prob = -0.5
        return action

    def log_prob(self, state: str, action: str) -> float:
        if self._cache_key == (state, action):
            return self._last_log_prob
        return -4.0

    def update(self, state: str, action: str, advantage: float, *, lr: float) -> None:
        return None
