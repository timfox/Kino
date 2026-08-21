"""SE(2)-style ego-motion operations (Sec. 2.1, Appendix A.2–A.4)."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor

Action = tuple[float, float, float]  # (dx, dy, dtheta)


def zero_action() -> Action:
    return (0.0, 0.0, 0.0)


def negate_action(action: Action) -> Action:
    """Operational inverse a^{-1} = -a under normalized increments (Appendix B.3)."""
    dx, dy, dt = action
    return (-dx, -dy, -dt)


def accumulate_actions(actions: Sequence[Action]) -> Action:
    """Sum normalized increments (local composition proxy, Eq. 41)."""
    sx = sy = st = 0.0
    for dx, dy, dt in actions:
        sx += dx
        sy += dy
        st += dt
    return (sx, sy, st)


def compose_action_segments(a: Sequence[Action], b: Sequence[Action]) -> list[Action]:
    """Concatenate two action segments (rollout ordering)."""
    return list(a) + list(b)


def inverse_segment(forward: Sequence[Action]) -> list[Action]:
    """Forward–inverse cycle u_inv (Eq. 16, 40)."""
    rev = [negate_action(x) for x in reversed(forward)]
    return list(forward) + rev


def composition_alternative_segment(
    segment: Sequence[Action],
    *,
    weights: Sequence[float] | None = None,
) -> list[Action]:
    """Eq. (17): redistribute cumulative increments with Dirichlet weights."""
    l = len(segment)
    if l == 0:
        return []
    if weights is None:
        w = torch.ones(l) / l
    else:
        w = torch.tensor(weights, dtype=torch.float32)
        w = w / w.sum().clamp(min=1e-8)
    total = accumulate_actions(segment)
    tx, ty, tt = total
    out: list[Action] = []
    for wi in w.tolist():
        out.append((wi * tx, wi * ty, wi * tt))
    return out


def actions_to_tensor(actions: Sequence[Action]) -> Tensor:
    """(T, 3) tensor."""
    if not actions:
        return torch.zeros(0, 3)
    return torch.tensor(actions, dtype=torch.float32)
