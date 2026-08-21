"""Backward search: goal tree scoring (Eq. 5–6)."""

from __future__ import annotations

from ltx_trainer.bes.types import GoalTree, Trajectory, VerifierFn


def backward_score(
    trajectory: Trajectory,
    goal_id: str,
    tree: GoalTree,
    verifiers: dict[str, VerifierFn],
    *,
    alpha: float,
    cache: dict[tuple[str, str], float] | None = None,
) -> float:
    """Recursive sub-goal score s(n, g) — Algorithm 3."""
    key = (goal_id, "|".join(trajectory))
    if cache is not None and key in cache:
        return cache[key]

    v_g = verifiers[goal_id](trajectory)
    if v_g >= 1.0 - 1e-9:
        score = 1.0
    else:
        children = tree.children_of(goal_id)
        if children:
            child_scores = [
                backward_score(
                    trajectory, c.goal_id, tree, verifiers, alpha=alpha, cache=cache
                )
                for c in children
            ]
            mean_child = sum(child_scores) / len(child_scores)
            score = alpha * v_g + (1.0 - alpha) * mean_child
        else:
            score = v_g

    if cache is not None:
        cache[key] = score
    return score


def pair_score(
    ta: Trajectory,
    tb: Trajectory,
    goal_id: str,
    tree: GoalTree,
    verifiers: dict[str, VerifierFn],
    *,
    alpha: float,
    cache: dict[tuple[str, str, str], float] | None = None,
) -> float:
    """Pair score s(na, nb, g) — Eq. 6."""
    key = (goal_id, "|".join(ta), "|".join(tb))
    if cache is not None and key in cache:
        return cache[key]

    va = verifiers[goal_id](ta)
    vb = verifiers[goal_id](tb)
    v_max = max(va, vb)
    if v_max >= 1.0 - 1e-9:
        score = 1.0
    else:
        children = tree.children_of(goal_id)
        if children:
            child_scores = [
                pair_score(ta, tb, c.goal_id, tree, verifiers, alpha=alpha, cache=cache)
                for c in children
            ]
            mean_child = sum(child_scores) / len(child_scores)
            score = alpha * v_max + (1.0 - alpha) * mean_child
        else:
            score = v_max

    if cache is not None:
        cache[key] = score
    return score


def score_node(
    trajectory: Trajectory, tree: GoalTree, verifiers: dict[str, VerifierFn], *, alpha: float
) -> float:
    return backward_score(trajectory, tree.root_id, tree, verifiers, alpha=alpha)


def score_pair(
    ta: Trajectory,
    tb: Trajectory,
    tree: GoalTree,
    verifiers: dict[str, VerifierFn],
    *,
    alpha: float,
) -> float:
    return pair_score(ta, tb, tree.root_id, tree, verifiers, alpha=alpha)
