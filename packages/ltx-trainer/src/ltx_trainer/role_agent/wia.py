"""World-In-Agent: predictive rewards and modulated returns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ltx_trainer.role_agent.lms_similarity import lms_similarity


@dataclass(frozen=True)
class StepRecord:
    state: str
    action: str
    reward: float


def predictive_reward_matrix(
    predictions: dict[int, dict[int, str]],
    states: Sequence[str],
    *,
    horizon: int,
    gamma: float = 0.99,
) -> list[list[float]]:
    """Compute r̃_{t,h} = LMS(ŝ_{t,h}, s_{t+h}) for t=0..T-1."""
    t_len = len(states)
    matrix: list[list[float]] = []
    for t in range(t_len):
        row: list[float] = []
        for h in range(1, horizon + 1):
            target_idx = t + h
            if target_idx >= t_len:
                row.append(0.0)
                continue
            pred = predictions.get(t, {}).get(h, "")
            row.append(lms_similarity(pred, states[target_idx]))
        matrix.append(row)
    return matrix


def task_return(rewards: Sequence[float], t: int, *, gamma: float) -> float:
    """Discounted return R_task(a_t) from step t (Eq. 5)."""
    return sum(gamma ** (k - t) * rewards[k] for k in range(t, len(rewards)))


def predictive_aggregate(row: Sequence[float], *, gamma: float) -> float:
    """R_pre(a_t) = Σ_h γ^{h-1} r̃_{t,h}."""
    return sum((gamma ** (h - 1)) * row[h - 1] for h in range(1, len(row) + 1))


def modulated_return(
    task: float,
    predictive: float,
) -> float:
    """R_t = R_task · (1 + R_pre); multiplicative modulation (Eq. 6)."""
    return task * (1.0 + predictive)


def compute_step_returns(
    trajectory: Sequence[StepRecord],
    predictions: dict[int, dict[int, str]],
    *,
    horizon: int,
    gamma: float = 0.99,
) -> list[float]:
    states = [s.state for s in trajectory]
    rewards = [s.reward for s in trajectory]
    pred_matrix = predictive_reward_matrix(predictions, states, horizon=horizon, gamma=gamma)
    out: list[float] = []
    for t, row in enumerate(pred_matrix):
        r_task = task_return(rewards, t, gamma=gamma)
        r_pre = predictive_aggregate(row, gamma=gamma)
        out.append(modulated_return(r_task, r_pre))
    return out
