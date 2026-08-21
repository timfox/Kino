"""Rollout collection with WIA predictions (Algorithm 1 lines 4–12)."""

from __future__ import annotations

from ltx_trainer.role_agent.envs.protocol import AgentEnv
from ltx_trainer.role_agent.config import RoleAgentConfig
from ltx_trainer.role_agent.policy import Policy
from ltx_trainer.role_agent.state_predictor import collect_predictions, wia_rewards_for_rollout
from ltx_trainer.role_agent.types import RolloutBatch, RolloutTrajectory
from ltx_trainer.role_agent.wia import StepRecord


def rollout_one(
    env: AgentEnv,
    policy: Policy,
    *,
    cfg: RoleAgentConfig,
    predict_fn=None,
) -> RolloutTrajectory:
    active = policy
    if hasattr(policy, "with_task"):
        active = policy.with_task(task_prompt=env.task.prompt, domain=env.task.domain)
    state = env.reset()
    steps: list[StepRecord] = []
    total = 0.0
    while not env.done:
        valid = env.valid_actions()
        action = active.choose_action(state, valid_actions=valid or None)
        next_state, reward, done = env.step_action(action)
        steps.append(StepRecord(state=state, action=action, reward=reward))
        total += reward
        state = next_state
        if done:
            break
    horizon = cfg.prediction_horizon(env.task.domain)
    predictions = collect_predictions(steps, horizon=horizon, predict_fn=predict_fn)
    matrix, step_returns = wia_rewards_for_rollout(steps, predictions, cfg=cfg, domain=env.task.domain)
    return RolloutTrajectory(
        task=env.task,
        steps=steps,
        predictions=predictions,
        success=env.success,
        total_return=total if total > 0 else step_returns[-1] if step_returns else 0.0,
        step_returns=step_returns,
        metadata={"predictive_matrix_rows": len(matrix), "backend": getattr(env, "backend", "toy")},
    )


def collect_rollouts(
    envs: list[AgentEnv],
    policy: Policy,
    *,
    cfg: RoleAgentConfig,
    group_size: int | None = None,
    predict_fn=None,
) -> RolloutBatch:
    gs = group_size or cfg.group_size
    trajectories: list[RolloutTrajectory] = []
    for env in envs:
        for _ in range(gs):
            trajectories.append(rollout_one(env, policy, cfg=cfg, predict_fn=predict_fn))
    return RolloutBatch(trajectories=trajectories)
