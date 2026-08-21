"""GiGPO-style state grouping and mixed trajectory/state advantages."""

from __future__ import annotations

import hashlib
import statistics
from collections import defaultdict
from typing import Sequence

from ltx_trainer.role_agent.lms_similarity import states_equivalent


def state_hash(state: str) -> str:
    return hashlib.sha256(state.strip().encode()).hexdigest()[:16]


def group_states(
    state_action_returns: Sequence[tuple[str, str, float]],
    *,
    similarity_threshold: float = 0.9,
) -> dict[str, list[tuple[str, str, float]]]:
    """Group (state, action, return) by LMS-equivalent states (Eq. 7)."""
    groups: dict[str, list[tuple[str, str, float]]] = {}
    canonical: dict[str, str] = {}
    for state, action, ret in state_action_returns:
        matched_key: str | None = None
        for key, rep in canonical.items():
            if states_equivalent(state, rep, threshold=similarity_threshold):
                matched_key = key
                break
        if matched_key is None:
            matched_key = state_hash(state)
            canonical[matched_key] = state
        groups.setdefault(matched_key, []).append((state, action, ret))
    return groups


def state_level_advantage(returns: Sequence[float]) -> list[float]:
    """Normalize returns within a state group (Eq. 8)."""
    if len(returns) <= 1:
        return [0.0 for _ in returns]
    mu = statistics.mean(returns)
    std = statistics.pstdev(returns)
    if std < 1e-8:
        return [0.0 for _ in returns]
    return [(r - mu) / std for r in returns]


def trajectory_level_advantage(rollout_returns: Sequence[float]) -> list[float]:
    """GRPO-style trajectory advantage A^E (Eq. 1)."""
    if len(rollout_returns) <= 1:
        return [0.0 for _ in rollout_returns]
    mu = statistics.mean(rollout_returns)
    std = statistics.pstdev(rollout_returns)
    if std < 1e-8:
        return [0.0 for _ in rollout_returns]
    return [(r - mu) / std for r in rollout_returns]


def mixed_advantages(
    step_returns: Sequence[float],
    states: Sequence[str],
    actions: Sequence[str],
    rollout_total: float,
    rollout_totals: Sequence[float],
    *,
    similarity_threshold: float = 0.9,
    alpha: float = 1.0,
) -> list[float]:
    """A = A^S + α·A^E per action step."""
    items = list(zip(states, actions, step_returns, strict=True))
    groups = group_states(items, similarity_threshold=similarity_threshold)
    state_adv_map: dict[tuple[str, str], float] = {}
    for members in groups.values():
        rets = [m[2] for m in members]
        advs = state_level_advantage(rets)
        for (st, act, _), adv in zip(members, advs, strict=True):
            state_adv_map[(st, act)] = adv
    traj_adv = trajectory_level_advantage(list(rollout_totals))
    rollout_idx = list(rollout_totals).index(rollout_total)
    a_e = traj_adv[rollout_idx]
    return [state_adv_map.get((s, a), 0.0) + alpha * a_e for s, a in zip(states, actions, strict=True)]
