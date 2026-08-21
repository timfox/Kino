"""Toy agent rollouts with latent vs in-context skills (ALFWorld replay fixtures)."""
from __future__ import annotations

from typing import Any

from ltx_trainer.latentskill.config import LatentSkillConfig
from ltx_trainer.latentskill.efficiency import compare_modes_on_history
from ltx_trainer.latentskill.inference import LatentSkillSession, choose_action_stub, choose_search_qa_action
from ltx_trainer.latentskill.skills import match_alfworld_skill, match_search_qa_skill


def run_replay_rollout(
    env,
    *,
    skill_name: str,
    mode: str = "latent",
    cfg: LatentSkillConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or LatentSkillConfig()
    session = LatentSkillSession(cfg)
    if mode == "latent":
        session.load_skill(skill_name)
    obs = env.reset()
    total_prefill_k = 0.0
    steps = 0
    is_search = getattr(env, "backend", "") == "search_qa"
    while not env.done and steps < env._max_steps:  # noqa: SLF001
        ctx = session.turn(obs, skill_name, mode=mode)
        total_prefill_k += ctx.estimate_prefill_tokens() / 1000.0
        valid = env.valid_actions()
        if is_search:
            action = choose_search_qa_action(valid, step=steps + 1, mode=mode)
        else:
            action = choose_action_stub(obs, valid, skill_name, mode=mode)
        obs, _reward, done = env.step_action(action)
        steps += 1
        if done:
            break
    return {
        "mode": mode,
        "skill": skill_name,
        "success": env.success,
        "steps": steps,
        "prefill_k": round(total_prefill_k, 3),
        "skill_tokens_in_prompt": mode == "in_context",
    }


def run_alfworld_fixture_compare(fixture_name: str = "alfworld_put_apple", *, task_type: str = "Clean") -> dict[str, Any]:
    from ltx_trainer.role_agent.envs.replay import make_replay_env

    skill_name = match_alfworld_skill(task_type)
    env_latent = make_replay_env(fixture_name)
    env_inctx = make_replay_env(fixture_name)
    latent = run_replay_rollout(env_latent, skill_name=skill_name, mode="latent")
    inctx = run_replay_rollout(env_inctx, skill_name=skill_name, mode="in_context")
    eff = compare_modes_on_history(env_latent._start, skill_name)  # noqa: SLF001
    return {
        "fixture": fixture_name,
        "skill": skill_name,
        "latent": latent,
        "in_context": inctx,
        "token_compare": eff,
        "latent_zero_skill_tokens": not latent["skill_tokens_in_prompt"],
    }


def run_search_qa_compare(task_id: str = "search_capital_france", *, dataset: str = "NQ") -> dict[str, Any]:
    from ltx_trainer.role_agent.envs.search_qa_env import SearchQAEnv

    skill_name = match_search_qa_skill(dataset)
    env_latent = SearchQAEnv.create(task_id)
    env_inctx = SearchQAEnv.create(task_id)
    latent = run_replay_rollout(env_latent, skill_name=skill_name, mode="latent")
    inctx = run_replay_rollout(env_inctx, skill_name=skill_name, mode="in_context")
    history = env_latent.task.prompt
    eff = compare_modes_on_history(history, skill_name)
    return {
        "task_id": task_id,
        "dataset": dataset,
        "skill": skill_name,
        "latent": latent,
        "in_context": inctx,
        "token_compare": eff,
        "latent_zero_skill_tokens": not latent["skill_tokens_in_prompt"],
    }


def rollout_demo(cfg: LatentSkillConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LatentSkillConfig()
    try:
        alf = run_alfworld_fixture_compare()
    except Exception as exc:  # noqa: BLE001
        alf = {"error": str(exc)}
    try:
        search = run_search_qa_compare()
    except Exception as exc:  # noqa: BLE001
        search = {"error": str(exc)}
    return {
        "alfworld_replay": alf,
        "search_qa": search,
        "cfg_alpha": cfg.default_injection_alpha,
    }
