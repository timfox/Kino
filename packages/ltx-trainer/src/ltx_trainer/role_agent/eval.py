"""Evaluate Role-Agent on toy text environments."""

from __future__ import annotations

from typing import Any

from ltx_trainer.role_agent.config import RoleAgentConfig
from ltx_trainer.role_agent.envs.text_toy import TOY_TASK_POOL, make_toy_env
from ltx_trainer.role_agent.policy import LLMActionPolicy, ScriptedPolicy, ToyLogPolicy
from ltx_trainer.role_agent.rollout import rollout_one


def evaluate_toy_tasks(policy, *, cfg: RoleAgentConfig | None = None) -> dict[str, float]:
    cfg = cfg or RoleAgentConfig()
    rates: dict[str, float] = {}
    for task_id in TOY_TASK_POOL:
        env = make_toy_env(task_id)
        traj = rollout_one(env, policy, cfg=cfg)
        rates[task_id] = 1.0 if traj.success else 0.0
    return rates


def evaluate_scripted_oracle(*, cfg: RoleAgentConfig | None = None) -> dict[str, float]:
    """Upper-bound scripted policies per task graph."""
    cfg = cfg or RoleAgentConfig()
    out: dict[str, float] = {}
    for task_id, template in TOY_TASK_POOL.items():
        plan: dict[str, str] = {}
        env = make_toy_env(task_id)
        state = env._start
        visited: set[str] = set()
        while state not in visited and state in env._graph:
            visited.add(state)
            actions = env._graph[state]
            for action, (nxt, note) in actions.items():
                if note in ("success", "ok"):
                    plan[state] = action
                    state = nxt
                    break
            else:
                break
        policy = ScriptedPolicy(plan=plan)
        traj = rollout_one(env, policy, cfg=cfg)
        out[task_id] = 1.0 if traj.success else 0.0
    return out


def evaluate_trained_policy(policy: ToyLogPolicy, *, cfg: RoleAgentConfig | None = None) -> dict[str, object]:
    rates = evaluate_toy_tasks(policy, cfg=cfg)
    return {
        "per_task": rates,
        "mean_success": sum(rates.values()) / max(len(rates), 1),
    }


def run_llm_rollout_demo(
    *,
    use_wia_llm: bool = True,
    client=None,
    backend: str | None = None,
) -> dict[str, Any]:
    """Evaluate tasks with vLLM action selection (+ optional WIA LLM preds)."""
    from ltx_trainer.role_agent.envs.registry import list_task_ids, make_env, resolve_backend
    from ltx_trainer.role_agent.llm_backend import (
        RoleAgentLLMClient,
        RoleAgentLLMConfig,
        make_vllm_state_predictor,
        role_agent_wia_llm_enabled,
        try_make_state_predictor,
    )

    cfg = RoleAgentConfig()
    mode = resolve_backend(backend)
    llm = client or RoleAgentLLMClient(RoleAgentLLMConfig.from_env())
    policy = LLMActionPolicy(client=llm)
    predict_fn = None
    if use_wia_llm and role_agent_wia_llm_enabled():
        predict_fn = try_make_state_predictor() or make_vllm_state_predictor(llm)
    per_task: dict[str, dict[str, Any]] = {}
    for task_id in list_task_ids(mode):
        env = make_env(task_id, mode)
        traj = rollout_one(env, policy, cfg=cfg, predict_fn=predict_fn)
        per_task[task_id] = {
            "success": traj.success,
            "steps": len(traj.steps),
            "domain": env.task.domain,
            "backend": traj.metadata.get("backend", mode),
            "actions": [s.action for s in traj.steps],
        }
    successes = sum(1 for v in per_task.values() if v["success"])
    return {
        "ok": True,
        "backend": mode,
        "model": llm.config.model,
        "wia_llm": predict_fn is not None,
        "tasks": len(per_task),
        "successes": successes,
        "success_rate": successes / max(len(per_task), 1),
        "per_task": per_task,
    }
