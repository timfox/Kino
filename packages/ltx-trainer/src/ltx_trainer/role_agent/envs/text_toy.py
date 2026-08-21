"""Deterministic mini ALFWorld / WebShop / Search-QA text environments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ltx_trainer.role_agent.types import TaskSpec


@dataclass
class TextToyEnv:
    """Graph-style text env with sparse terminal reward (suc=1, fail=0)."""

    task: TaskSpec
    state: str = ""
    step: int = 0
    done: bool = False
    success: bool = False
    _graph: dict[str, dict[str, tuple[str, str]]] = field(default_factory=dict)
    _start: str = ""
    _max_steps: int = 50

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


def _alfworld_put_apple_fridge() -> TextToyEnv:
    task = TaskSpec(
        task_id="alf_put_apple",
        domain="alfworld",
        prompt="put a clean apple in the fridge",
        objective="Navigate, take apple, clean if needed, place in fridge",
    )
    env = TextToyEnv(task=task, _max_steps=8)
    env._start = "You are in the kitchen. You see apple 1 on counter 1 and fridge 1."
    env._graph = {
        env._start: {
            "go to counter 1": (
                "On counter 1 you see apple 1 and knife 1.",
                "ok",
            ),
            "take apple 1 from counter 1": (
                env._start,
                "invalid",
            ),
        },
        "On counter 1 you see apple 1 and knife 1.": {
            "take apple 1 from counter 1": (
                "You pick up apple 1.",
                "ok",
            ),
            "go to fridge 1": (
                "The fridge 1 is closed.",
                "ok",
            ),
        },
        "You pick up apple 1.": {
            "go to sinkbasin 1": (
                "At sinkbasin 1. You can clean apple 1.",
                "ok",
            ),
            "go to fridge 1": (
                "The fridge 1 is closed. Apple is not clean.",
                "fail",
            ),
        },
        "At sinkbasin 1. You can clean apple 1.": {
            "clean apple 1 with sinkbasin 1": (
                "Apple 1 is clean.",
                "ok",
            ),
        },
        "Apple 1 is clean.": {
            "go to fridge 1": (
                "The fridge 1 is closed.",
                "ok",
            ),
        },
        "The fridge 1 is closed.": {
            "open fridge 1": (
                "Fridge 1 is open.",
                "ok",
            ),
        },
        "Fridge 1 is open.": {
            "put apple 1 in fridge 1": (
                "You put apple 1 in fridge 1. Task success.",
                "success",
            ),
        },
    }
    return env


def _alfworld_wrong_receptacle() -> TextToyEnv:
    task = TaskSpec(
        task_id="alf_wrong_bin",
        domain="alfworld",
        prompt="put apple in garbage can",
        objective="Place apple in garbage can",
    )
    env = TextToyEnv(task=task, _max_steps=6)
    env._start = "Kitchen. apple 1 on counter. garbagecan 1 and fridge 1 visible."
    env._graph = {
        env._start: {
            "take apple 1 from counter 1": ("You pick up apple 1.", "ok"),
        },
        "You pick up apple 1.": {
            "go to garbagecan 1": ("At garbagecan 1.", "ok"),
            "go to fridge 1": ("At fridge 1.", "ok"),
        },
        "At garbagecan 1.": {
            "put apple 1 in garbagecan 1": ("Apple in garbagecan. Task success.", "success"),
        },
        "At fridge 1.": {
            "put apple 1 in fridge 1": ("Wrong receptacle. Task failed.", "fail"),
        },
    }
    return env


def _webshop_buy_mug() -> TextToyEnv:
    task = TaskSpec(
        task_id="web_mug_blue",
        domain="webshop",
        prompt="Buy a blue ceramic mug under $20",
        objective="Search, select matching product, purchase",
    )
    env = TextToyEnv(task=task, _max_steps=5)
    env._start = "WebShop home. Search bar ready."
    env._graph = {
        env._start: {
            "search[blue ceramic mug]": (
                "Results: [1] Blue Ceramic Mug $14.99 [2] Red Mug $9.99",
                "ok",
            ),
        },
        "Results: [1] Blue Ceramic Mug $14.99 [2] Red Mug $9.99": {
            "click[1]": ("Product page: Blue Ceramic Mug $14.99", "ok"),
            "click[2]": ("Product page: Red Mug $9.99", "ok"),
        },
        "Product page: Blue Ceramic Mug $14.99": {
            "click[buy now]": ("Purchase complete. Task success.", "success"),
        },
        "Product page: Red Mug $9.99": {
            "click[buy now]": ("Wrong color. Task failed.", "fail"),
        },
    }
    return env


def _search_qa_hop() -> TextToyEnv:
    task = TaskSpec(
        task_id="search_capital_france",
        domain="search_qa",
        prompt="What is the capital of France?",
        objective="Multi-hop search then answer",
    )
    env = TextToyEnv(task=task, _max_steps=4)
    env._start = "Search agent ready. No history yet."
    env._graph = {
        env._start: {
            "search[france capital]": (
                "Doc: Paris is the capital and largest city of France.",
                "ok",
            ),
            "answer[lyon]": ("Incorrect answer.", "fail"),
        },
        "Doc: Paris is the capital and largest city of France.": {
            "answer[paris]": ("Correct. Task success.", "success"),
            "search[france]": ("Doc: France is in Western Europe.", "ok"),
        },
        "Doc: France is in Western Europe.": {
            "answer[paris]": ("Correct. Task success.", "success"),
        },
    }
    return env


TOY_TASK_POOL: dict[str, TextToyEnv] = {
    "alf_put_apple": _alfworld_put_apple_fridge(),
    "alf_wrong_bin": _alfworld_wrong_receptacle(),
    "web_mug_blue": _webshop_buy_mug(),
    "search_capital_france": _search_qa_hop(),
}


def make_toy_env(task_id: str) -> TextToyEnv:
    if task_id not in TOY_TASK_POOL:
        raise KeyError(f"unknown toy task: {task_id}")
    base = TOY_TASK_POOL[task_id]
    return TextToyEnv(
        task=base.task,
        _graph=dict(base._graph),
        _start=base._start,
        _max_steps=base._max_steps,
    )


def domain_task_ids(domain: str) -> list[str]:
    dom = domain.lower()
    return [tid for tid, env in TOY_TASK_POOL.items() if env.task.domain == dom or (dom == "search" and env.task.domain == "search_qa")]
