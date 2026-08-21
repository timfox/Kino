"""Recovered-state distance and pose helpers (Eq. 7–8)."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor

State = tuple[Tensor, Tensor]  # (translation (2,), rotation scalar or matrix)


def state_from_xytheta(x: float, y: float, theta: float) -> State:
    return (torch.tensor([x, y], dtype=torch.float32), torch.tensor(theta, dtype=torch.float32))


def rotation_distance(r1: Tensor, r2: Tensor) -> float:
    """Angular discrepancy for planar heading (radians)."""
    if r1.numel() == 1 and r2.numel() == 1:
        d = abs(float(r1.item() - r2.item()))
        return min(d, 2 * math.pi - d)
    # 2x2 rotation matrices: Frobenius norm proxy
    return float(torch.linalg.norm(r1 - r2).item())


def state_distance(
    s1: State,
    s2: State,
    *,
    alpha: float = 1.0,
) -> float:
    """Eq. (8): d(si, sj) = ||pi - pj||_2 + alpha * d_R(Ri, Rj)."""
    p1, r1 = s1
    p2, r2 = s2
    trans = float(torch.linalg.norm(p1 - p2).item())
    rot = rotation_distance(r1, r2)
    return trans + alpha * rot


def trajectory_gar_distance(
    trajectories: Sequence[Sequence[State]],
    *,
    alpha: float = 1.0,
) -> float:
    """Eq. (21): mean pairwise state distance over time (EGAR building block)."""
    n = len(trajectories)
    if n < 2:
        return 0.0
    t_len = min(len(tr) for tr in trajectories)
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            for t in range(t_len):
                total += state_distance(trajectories[i][t], trajectories[j][t], alpha=alpha)
            pairs += t_len
    return total / max(pairs, 1)
