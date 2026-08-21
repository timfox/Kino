"""Toy text environments for Role-Agent CPU training."""

from ltx_trainer.role_agent.envs.registry import (
    env_catalog,
    list_task_ids,
    make_env,
    probe_all_envs,
    resolve_backend,
)
from ltx_trainer.role_agent.envs.replay import ReplayTraceEnv, make_replay_env
from ltx_trainer.role_agent.envs.text_toy import (
    TOY_TASK_POOL,
    TextToyEnv,
    make_toy_env,
)

__all__ = [
    "TOY_TASK_POOL",
    "ReplayTraceEnv",
    "TextToyEnv",
    "env_catalog",
    "list_task_ids",
    "make_env",
    "make_replay_env",
    "make_toy_env",
    "probe_all_envs",
    "resolve_backend",
]
