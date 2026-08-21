"""Bidirectional Evolutionary Search — Algorithm 1."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field

from ltx_trainer.bes.backward import score_node
from ltx_trainer.bes.config import BESConfig
from ltx_trainer.bes.forward import forward_step
from ltx_trainer.bes.types import Goal, GoalTree, Node, Trajectory, VerifierFn


@dataclass
class BESSearchResult:
    """Outcome of one BES run on a single problem."""

    best_trajectory: Trajectory | None
    best_score: float
    terminal_pool: list[Trajectory]
    policy_calls: int
    pool_size: int
    found_success: bool
    trace: list[str] = field(default_factory=list)


def _eligible_indices(
    pool: list[Node],
    terminal_check: Callable[[Trajectory], bool],
) -> list[int]:
    return [i for i, n in enumerate(pool) if not terminal_check(n.steps)]


def backward_decompose(
    tree: GoalTree,
    verifiers: dict[str, VerifierFn],
    pool: list[Node],
    decompose_fn: Callable[[Goal, list[Node]], list[tuple[str, str, VerifierFn]]],
    rng: random.Random,
) -> None:
    """Algorithm 4 — refine one unsolved leaf."""
    leaves = tree.leaves()
    unsolved: list[Goal] = []
    for g in leaves:
        best = max(verifiers[g.goal_id](n.steps) for n in pool) if pool else 0.0
        if best < 1.0 - 1e-9:
            unsolved.append(g)
    if not unsolved:
        return
    g_star = unsolved[rng.randint(0, len(unsolved) - 1)]
    children_spec = decompose_fn(g_star, pool)
    for cid, desc, vf in children_spec:
        child = Goal(goal_id=cid, description=desc, parent_id=g_star.goal_id)
        tree.goals[cid] = child
        verifiers[cid] = vf
        g_star.children.append(cid)


def bidirectional_evolutionary_search(
    *,
    policy: Callable[[Trajectory, int], list[str]],
    terminal_verifier: VerifierFn,
    terminal_check: Callable[[Trajectory], bool],
    initial_tree: GoalTree,
    initial_verifiers: dict[str, VerifierFn],
    decompose_fn: Callable[[Goal, list[Node]], list[tuple[str, str, VerifierFn]]],
    cfg: BESConfig | None = None,
    seed: int = 0,
) -> BESSearchResult:
    """
    Run BES on one problem instance.

    ``terminal_verifier`` is V(x, n) for the root goal; ``initial_verifiers`` must
    include the root id with the same function.
    """
    cfg = cfg or BESConfig()
    rng = random.Random(seed)
    tree = initial_tree
    verifiers = dict(initial_verifiers)

    pool: list[Node] = [Node(())]
    scores: dict[int, float] = {0: 0.0}
    parent_degree: dict[int, int] = {}
    policy_calls = 0
    trace: list[str] = []

    for t in range(cfg.budget_calls):
        eligible = _eligible_indices(pool, terminal_check)
        if not eligible:
            break

        child = forward_step(
            pool,
            eligible,
            tree,
            verifiers,
            operator_probs=cfg.operator_probs,
            policy=policy,
            k_max=cfg.k_max_expand,
            step_idx=t,
            budget=cfg.budget_calls,
            tau_start=cfg.tau_start,
            tau_end=cfg.tau_end,
            unexplored_bonus=cfg.unexplored_bonus,
            parent_degree=parent_degree,
            rng=rng,
            alpha=cfg.alpha,
        )
        if child is None:
            continue

        # Count policy calls on expansion only (approximate)
        if len(child.steps) > 0:
            policy_calls += 1

        idx = len(pool)
        pool.append(child)
        scores[idx] = score_node(child.steps, tree, verifiers, alpha=cfg.alpha)

        if (t + 1) % cfg.decompose_interval == 0:
            backward_decompose(tree, verifiers, pool, decompose_fn, rng)
            for i, n in enumerate(pool):
                scores[i] = score_node(n.steps, tree, verifiers, alpha=cfg.alpha)
            trace.append(f"decompose@step={t + 1}")

        if terminal_check(child.steps) and terminal_verifier(child.steps) >= 1.0 - 1e-9:
            trace.append(f"success@step={t + 1}")
            return BESSearchResult(
                best_trajectory=child.steps,
                best_score=1.0,
                terminal_pool=[n.steps for n in pool if terminal_check(n.steps)],
                policy_calls=policy_calls,
                pool_size=len(pool),
                found_success=True,
                trace=trace,
            )

    terminals = [n.steps for n in pool if terminal_check(n.steps)]
    if terminals:
        best_traj = max(terminals, key=terminal_verifier)
        best_sc = terminal_verifier(best_traj)
    else:
        best_idx = max(scores, key=lambda i: scores[i])
        best_traj = pool[best_idx].steps
        best_sc = scores[best_idx]

    return BESSearchResult(
        best_trajectory=best_traj if best_traj else None,
        best_score=best_sc,
        terminal_pool=terminals,
        policy_calls=policy_calls,
        pool_size=len(pool),
        found_success=best_sc >= 1.0 - 1e-9,
        trace=trace,
    )
