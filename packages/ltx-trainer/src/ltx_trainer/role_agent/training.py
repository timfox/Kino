"""Algorithm 1 Role-Agent training loop (CPU toy + VeRL plan hook)."""

from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass, field
from typing import Any

from ltx_trainer.role_agent.aiw import (
    FailureMemory,
    classify_failure_heuristic,
    parse_reflection,
)
from ltx_trainer.role_agent.config import RoleAgentConfig
from ltx_trainer.role_agent.envs.registry import list_task_ids, make_env, resolve_backend, task_prompt_map
from ltx_trainer.role_agent.grpo import build_grpo_samples_from_rollouts, role_agent_grpo_loss
from ltx_trainer.role_agent.policy import ToyLogPolicy
from ltx_trainer.role_agent.retrieval import curriculum_resample_weights_retrieval
from ltx_trainer.role_agent.rollout import collect_rollouts


@dataclass
class TrainingState:
    iteration: int = 0
    memory: FailureMemory = field(default_factory=FailureMemory)
    task_weights: dict[str, float] = field(default_factory=dict)
    success_rate_history: list[float] = field(default_factory=list)
    loss_history: list[float] = field(default_factory=list)
    failure_mode_counts: dict[str, int] = field(default_factory=dict)


def _trajectory_text(traj) -> str:
    lines = []
    for i, step in enumerate(traj.steps, start=1):
        lines.append(f"Step {i} OBS: {step.state}")
        lines.append(f"Step {i} ACT: {step.action}")
    return "\n".join(lines)


def analyze_failures_aiw(
    batch,
    memory: FailureMemory,
    *,
    llm_client=None,
    use_llm_reflection: str | None = None,
) -> int:
    """Steps 17–20: failure analysis and memory update."""
    from ltx_trainer.role_agent.llm_backend import llm_analyze_failure, role_agent_aiw_llm_enabled

    added = 0
    for traj in batch.failed():
        text = _trajectory_text(traj)
        reflection = None
        if use_llm_reflection:
            reflection = parse_reflection(use_llm_reflection, task_id=traj.task.task_id, domain=traj.task.domain)
        elif llm_client is not None and role_agent_aiw_llm_enabled():
            try:
                reflection = llm_analyze_failure(
                    llm_client,
                    task=traj.task.prompt,
                    trajectory=text,
                    task_id=traj.task.task_id,
                    domain=traj.task.domain,
                )
            except Exception:
                reflection = None
        if reflection is None:
            mode = classify_failure_heuristic(text, traj.task.domain)
            reflection = parse_reflection(
                f"""<reflection>
DOMINANT_TYPE: {mode}
CORE_LESSON: Practice {mode.replace('_', ' ')} patterns.
RETRIEVAL_QUERY: {mode} {traj.task.prompt[:40]}
</reflection>""",
                task_id=traj.task.task_id,
                domain=traj.task.domain,
            )
        if reflection:
            memory.add(reflection)
            added += 1
    return added


def _record_failure_modes(memory: FailureMemory, counts: dict[str, int]) -> None:
    for entry in memory.entries:
        counts[entry.dominant_type] = counts.get(entry.dominant_type, 0) + 1


def update_task_distribution(
    task_ids: list[str],
    memory: FailureMemory,
    task_prompts: dict[str, str],
    *,
    boost: float = 2.0,
    llm_client=None,
    use_llm_retrieval: bool = False,
) -> dict[str, float]:
    """Steps 21–22: reshape p_D from failure memory + retrieval."""
    return curriculum_resample_weights_retrieval(
        task_ids,
        memory,
        task_prompts,
        boost=boost,
        llm_client=llm_client,
        use_llm_retrieval=use_llm_retrieval,
    )


def sample_tasks(task_ids: list[str], weights: dict[str, float], n: int, rng: random.Random) -> list[str]:
    if not task_ids:
        return []
    w = [weights.get(t, 1.0) for t in task_ids]
    total = sum(w)
    probs = [x / total for x in w]
    return rng.choices(task_ids, weights=probs, k=n)


def run_training_iteration(
    policy: ToyLogPolicy,
    ref_policy: ToyLogPolicy,
    state: TrainingState,
    *,
    cfg: RoleAgentConfig,
    task_ids: list[str] | None = None,
    group_size: int | None = None,
    lr: float = 0.05,
    rng: random.Random | None = None,
    predict_fn=None,
    llm_client=None,
    use_llm_retrieval: bool = False,
) -> dict[str, Any]:
    """One Algorithm 1 iteration on toy text envs."""
    rng = rng or random.Random(0)
    gs = group_size if group_size is not None else cfg.group_size
    pool = task_ids or list_task_ids(os.environ.get("GOPEX_ROLE_AGENT_ENV", "toy"))
    prompts = task_prompt_map(pool)
    if not state.task_weights:
        state.task_weights = {t: 1.0 for t in pool}
    chosen = sample_tasks(pool, state.task_weights, max(1, len(pool) // 2), rng)
    envs = [make_env(tid) for tid in chosen]
    batch = collect_rollouts(envs, policy, cfg=cfg, group_size=gs, predict_fn=predict_fn)
    samples = build_grpo_samples_from_rollouts(
        batch.trajectories,
        policy,
        ref_policy,
        alpha=cfg.advantage_alpha,
        similarity_threshold=cfg.state_similarity_threshold,
    )
    loss = role_agent_grpo_loss(
        samples,
        beta=cfg.kl_coefficient,
        epsilon_low=cfg.clip_ratio_low,
        epsilon_high=cfg.clip_ratio_high,
    )
    for s in samples:
        policy.update(s.state, s.action, s.advantage, lr=lr)
    added = 0
    if cfg.enable_aiw:
        added = analyze_failures_aiw(batch, state.memory, llm_client=llm_client)
        _record_failure_modes(state.memory, state.failure_mode_counts)
        state.task_weights = update_task_distribution(
            pool,
            state.memory,
            prompts,
            llm_client=llm_client,
            use_llm_retrieval=use_llm_retrieval,
        )
    elif not state.task_weights:
        state.task_weights = {t: 1.0 for t in pool}
    succ = sum(1 for t in batch.trajectories if t.success) / max(len(batch.trajectories), 1)
    state.iteration += 1
    state.success_rate_history.append(succ)
    state.loss_history.append(loss)
    return {
        "iteration": state.iteration,
        "loss": loss,
        "success_rate": succ,
        "failures_added": added,
        "memory_size": len(state.memory.entries),
        "unique_modes": len(state.memory.unique_modes()),
        "failure_mode_counts": dict(state.failure_mode_counts),
        "wia_enabled": cfg.enable_wia,
        "aiw_enabled": cfg.enable_aiw,
        "sampled_tasks": chosen,
    }


def run_toy_training(
    *,
    iterations: int = 20,
    seed: int = 42,
    group_size: int = 2,
    lr: float = 0.08,
    use_llm: bool | None = None,
    use_llm_retrieval: bool | None = None,
    enable_wia: bool = True,
    enable_aiw: bool = True,
    env_backend: str | None = None,
) -> dict[str, Any]:
    """Full CPU toy co-evolution smoke (WIA + AIW + GRPO)."""
    from ltx_trainer.role_agent.llm_backend import (
        RoleAgentLLMClient,
        RoleAgentLLMConfig,
        make_vllm_state_predictor,
        role_agent_aiw_llm_enabled,
        role_agent_llm_enabled,
        try_make_state_predictor,
    )

    cfg = RoleAgentConfig(enable_wia=enable_wia, enable_aiw=enable_aiw)
    if env_backend:
        os.environ["GOPEX_ROLE_AGENT_ENV"] = env_backend
    rng = random.Random(seed)
    policy = ToyLogPolicy(temperature=1.2)
    policy.seed(seed)
    ref = copy.deepcopy(policy)
    state = TrainingState()
    logs: list[dict[str, Any]] = []
    llm_client = None
    predict_fn = None
    if use_llm is None:
        use_llm = role_agent_llm_enabled()
    if use_llm_retrieval is None:
        use_llm_retrieval = role_agent_aiw_llm_enabled()
    if use_llm:
        try:
            llm_client = RoleAgentLLMClient(RoleAgentLLMConfig.from_env())
            predict_fn = try_make_state_predictor() or make_vllm_state_predictor(llm_client)
        except Exception:
            llm_client = None
            predict_fn = None
    for _ in range(iterations):
        logs.append(
            run_training_iteration(
                policy,
                ref,
                state,
                cfg=cfg,
                group_size=group_size,
                lr=lr,
                rng=rng,
                predict_fn=predict_fn,
                llm_client=llm_client,
                use_llm_retrieval=bool(use_llm_retrieval and llm_client),
            )
        )
    initial_sr = logs[0]["success_rate"] if logs else 0.0
    final_sr = logs[-1]["success_rate"] if logs else 0.0
    memory_ok = len(state.memory.entries) >= 1 if enable_aiw else True
    return {
        "ok": bool(logs) and memory_ok and all(
            isinstance(x["loss"], (int, float)) for x in logs
        ),
        "iterations": iterations,
        "llm_enabled": bool(llm_client),
        "llm_retrieval": bool(use_llm_retrieval and llm_client),
        "initial_success_rate": initial_sr,
        "final_success_rate": final_sr,
        "memory_entries": len(state.memory.entries),
        "unique_failure_modes": state.memory.unique_modes(),
        "failure_mode_evolution": dict(state.failure_mode_counts),
        "wia_enabled": enable_wia,
        "aiw_enabled": enable_aiw,
        "env_backend": env_backend or os.environ.get("GOPEX_ROLE_AGENT_ENV", "toy"),
        "last_loss": logs[-1]["loss"] if logs else 0.0,
        "logs": logs[-3:],
    }
