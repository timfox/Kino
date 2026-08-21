"""GRPO clipped objective with Role-Agent mixed advantages (Eq. 9)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ltx_trainer.role_agent.gigpo import mixed_advantages


@dataclass
class GrpoStepSample:
    state: str
    action: str
    log_prob: float
    old_log_prob: float
    ref_log_prob: float
    advantage: float
    rollout_idx: int


def kl_penalty(log_prob: float, ref_log_prob: float) -> float:
    """Low-variance KL estimator (same as glass/grpo)."""
    ratio = math.exp(ref_log_prob - log_prob)
    diff = ref_log_prob - log_prob
    return ratio - diff - 1.0


def clipped_policy_loss(
    log_prob: float,
    old_log_prob: float,
    advantage: float,
    *,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.28,
) -> float:
    ratio = math.exp(log_prob - old_log_prob)
    clipped = max(1.0 - epsilon_low, min(1.0 + epsilon_high, ratio))
    return -min(ratio * advantage, clipped * advantage)


def role_agent_grpo_loss(
    samples: list[GrpoStepSample],
    *,
    beta: float = 1e-3,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.28,
) -> float:
    if not samples:
        return 0.0
    total = 0.0
    for s in samples:
        pg = clipped_policy_loss(
            s.log_prob, s.old_log_prob, s.advantage, epsilon_low=epsilon_low, epsilon_high=epsilon_high
        )
        kl = beta * kl_penalty(s.log_prob, s.ref_log_prob)
        total += pg + kl
    return total / len(samples)


def build_grpo_samples_from_rollouts(
    trajectories: list,
    policy,
    ref_policy,
    *,
    alpha: float = 1.0,
    similarity_threshold: float = 0.9,
) -> list[GrpoStepSample]:
    """Attach mixed advantages and flatten to token/step samples."""
    episode_returns = [float(t.total_return) for t in trajectories]
    samples: list[GrpoStepSample] = []
    for i, traj in enumerate(trajectories):
        states = [s.state for s in traj.steps]
        actions = [s.action for s in traj.steps]
        step_returns = traj.step_returns or [traj.total_return] * len(actions)
        mixed = mixed_advantages(
            step_returns,
            states,
            actions,
            traj.total_return,
            episode_returns,
            alpha=alpha,
            similarity_threshold=similarity_threshold,
        )
        traj.mixed_advantages = mixed
        for t, step in enumerate(traj.steps):
            lp = policy.log_prob(step.state, step.action)
            old_lp = lp
            ref_lp = ref_policy.log_prob(step.state, step.action)
            samples.append(
                GrpoStepSample(
                    state=step.state,
                    action=step.action,
                    log_prob=lp,
                    old_log_prob=old_lp,
                    ref_log_prob=ref_lp,
                    advantage=mixed[t],
                    rollout_idx=i,
                )
            )
    return samples
