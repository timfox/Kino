"""Forward evolution operators (Appendix B) + expansion."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from ltx_trainer.bes.types import Node, Step, Trajectory

if TYPE_CHECKING:
    from collections.abc import Callable


def common_prefix_len(a: Trajectory, b: Trajectory) -> int:
    s = 0
    for ya, yb in zip(a, b, strict=False):
        if ya != yb:
            break
        s += 1
    return s


def expand_trajectory(
    parent: Trajectory,
    *,
    policy: Callable[[Trajectory, int], list[Step]],
    k_max: int,
    rng: random.Random,
) -> Trajectory:
    """Sample K ~ Uniform{1..Kmax} new steps (Eq. 2)."""
    k = rng.randint(1, max(1, k_max))
    steps = list(policy(parent, k))
    return parent + tuple(steps[:k])


def delete_step(parent: Trajectory, rng: random.Random) -> Trajectory | None:
    """Remove one interior step (Appendix B.ii)."""
    t = len(parent)
    if t < 3:
        return None
    ell = rng.randint(1, t - 2)  # 0-indexed interior: 2..t-1 in 1-based
    return parent[:ell] + parent[ell + 1 :]


def combine_trajectories(na: Trajectory, nb: Trajectory) -> Trajectory | None:
    """Concatenate suffixes beyond shared prefix (Appendix B.i)."""
    s = common_prefix_len(na, nb)
    if s == 0 and na and nb:
        return None
    sigma_a = na[s:]
    sigma_b = nb[s:]
    if not sigma_a or not sigma_b:
        return None
    return na[:s] + sigma_a + sigma_b


def translocate_trajectories(
    na: Trajectory, nb: Trajectory, rng: random.Random
) -> Trajectory | None:
    """Replace one step in A with a step from B (Appendix B.iii)."""
    s = common_prefix_len(na, nb)
    sigma_a = na[s:]
    sigma_b = nb[s:]
    ma, mb = len(sigma_a), len(sigma_b)
    if ma < 1 or mb < 1:
        return None
    r = rng.randint(0, ma - 1)
    q = rng.randint(0, mb - 1)
    repl = sigma_b[q]
    return na[:s] + sigma_a[:r] + (repl,) + sigma_a[r + 1 :]


def crossover_trajectories(
    na: Trajectory, nb: Trajectory, rng: random.Random
) -> Trajectory | None:
    """Splice prefix of A with tail of B (Appendix B.iv)."""
    s = common_prefix_len(na, nb)
    sigma_a = na[s:]
    sigma_b = nb[s:]
    ma, mb = len(sigma_a), len(sigma_b)
    if ma < 1 or mb < 1:
        return None
    i = rng.randint(0, ma)
    j = rng.randint(0, mb - 1)
    return na[:s] + sigma_a[:i] + sigma_b[j + 1 :]


def apply_operator(
    op: str,
    parent_a: Node,
    parent_b: Node | None,
    *,
    policy: Callable[[Trajectory, int], list[Step]],
    k_max: int,
    rng: random.Random,
) -> Node | None:
    """Apply one forward operator; return child node or None if invalid."""
    ya, yb = parent_a.steps, parent_b.steps if parent_b else ()
    if op == "expand":
        child = expand_trajectory(ya, policy=policy, k_max=k_max, rng=rng)
        return Node(child)
    if op == "delete":
        child = delete_step(ya, rng)
        return Node(child) if child else None
    if parent_b is None:
        return None
    if op == "combine":
        child = combine_trajectories(ya, yb)
    elif op == "translocate":
        child = translocate_trajectories(ya, yb, rng)
    elif op == "crossover":
        child = crossover_trajectories(ya, yb, rng)
    else:
        return None
    return Node(child) if child else None
