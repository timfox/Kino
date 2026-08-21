"""Unified environment registry: toy, replay, and optional live backends."""

from __future__ import annotations

import os
from typing import Any, Literal

from ltx_trainer.role_agent.envs.alfworld_adapter import AlfWorldLiveEnv, probe_alfworld
from ltx_trainer.role_agent.envs.replay import REPLAY_TASK_POOL, ReplayTraceEnv, list_replay_fixtures, make_replay_env
from ltx_trainer.role_agent.envs.search_adapter import probe_search_qa
from ltx_trainer.role_agent.envs.search_qa_env import SearchQAEnv, list_search_qa_task_ids, search_qa_task_prompts
from ltx_trainer.role_agent.envs.text_toy import TOY_TASK_POOL, TextToyEnv, make_toy_env
from ltx_trainer.role_agent.envs.webshop_adapter import probe_webshop
from ltx_trainer.role_agent.envs.webshop_live import WebShopLiveEnv

EnvBackend = Literal["auto", "toy", "replay", "alfworld", "webshop", "search_qa"]

# Map toy ids → replay fixture stems for admissible-command training shape
TOY_TO_REPLAY: dict[str, str] = {
    "alf_put_apple": "alfworld_put_apple",
    "web_mug_blue": "webshop_mug_blue",
    "search_capital_france": "search_capital_france",
}


def resolve_backend(backend: str | None = None) -> EnvBackend:
    raw = (backend or os.environ.get("GOPEX_ROLE_AGENT_ENV", "auto")).strip().lower()
    allowed = {"auto", "toy", "replay", "alfworld", "webshop", "search_qa"}
    if raw not in allowed:
        raise ValueError(f"unknown GOPEX_ROLE_AGENT_ENV={raw!r}; expected one of {sorted(allowed)}")
    return raw  # type: ignore[return-value]


def env_catalog() -> dict[str, Any]:
    return {
        "backends": ["auto", "toy", "replay", "alfworld", "webshop", "search_qa"],
        "toy_tasks": sorted(TOY_TASK_POOL.keys()),
        "search_qa_tasks": list_search_qa_task_ids(),
        "replay_fixtures": list_replay_fixtures(),
        "toy_to_replay": dict(TOY_TO_REPLAY),
        "probes": probe_all_envs(),
    }


def probe_all_envs() -> dict[str, Any]:
    return {
        "alfworld": probe_alfworld(),
        "webshop": probe_webshop(),
        "search_qa": probe_search_qa(),
    }


def make_env(
    task_id: str, backend: str | None = None
) -> TextToyEnv | ReplayTraceEnv | AlfWorldLiveEnv | SearchQAEnv | WebShopLiveEnv:
    """Create an environment by task id and backend (auto → live if ready else replay/toy)."""
    mode = resolve_backend(backend)
    if mode == "toy":
        return make_toy_env(task_id)
    if mode == "replay":
        fixture = TOY_TO_REPLAY.get(task_id, task_id)
        return make_replay_env(fixture)
    if mode == "alfworld":
        if AlfWorldLiveEnv.available():
            return AlfWorldLiveEnv.create()
        fixture = TOY_TO_REPLAY.get(task_id, "alfworld_put_apple")
        return make_replay_env(fixture)
    if mode == "webshop":
        return WebShopLiveEnv.create(task_id)
    if mode == "search_qa":
        if task_id in list_search_qa_task_ids():
            return SearchQAEnv.create(task_id)
        fixture = TOY_TO_REPLAY.get(task_id, "search_capital_france")
        return make_replay_env(fixture)

    # auto
    if task_id in list_search_qa_task_ids():
        return SearchQAEnv.create(task_id)
    domain = TOY_TASK_POOL[task_id].task.domain if task_id in TOY_TASK_POOL else task_id.split("_")[0]
    if domain == "alfworld" and AlfWorldLiveEnv.available():
        return AlfWorldLiveEnv.create()
    if task_id in TOY_TO_REPLAY:
        return make_replay_env(TOY_TO_REPLAY[task_id])
    if task_id in TOY_TASK_POOL:
        return make_toy_env(task_id)
    if task_id in REPLAY_TASK_POOL:
        return make_replay_env(task_id)
    raise KeyError(f"unknown task_id: {task_id}")


def list_task_ids(backend: str | None = None) -> list[str]:
    mode = resolve_backend(backend)
    if mode == "toy":
        return sorted(TOY_TASK_POOL.keys())
    if mode == "replay":
        return list_replay_fixtures()
    if mode == "search_qa":
        return list_search_qa_task_ids()
    return sorted(TOY_TASK_POOL.keys())


def task_prompt_map(task_ids: list[str] | None = None) -> dict[str, str]:
    """Task id → natural-language prompt for AIW retrieval."""
    ids = task_ids or list_task_ids("auto")
    prompts: dict[str, str] = {}
    for tid in ids:
        if tid in TOY_TASK_POOL:
            prompts[tid] = TOY_TASK_POOL[tid].task.prompt
        elif tid in search_qa_task_prompts():
            prompts[tid] = search_qa_task_prompts()[tid]
        elif tid in REPLAY_TASK_POOL:
            prompts[tid] = REPLAY_TASK_POOL[tid].task.prompt
        else:
            prompts[tid] = tid.replace("_", " ")
    return prompts
