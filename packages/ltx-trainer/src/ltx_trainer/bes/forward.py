"""Forward step: operator sampling + Boltzmann parent selection."""

from __future__ import annotations

import math
import random
from collections.abc import Callable

from ltx_trainer.bes.backward import score_node, score_pair
from ltx_trainer.bes.operators import apply_operator
from ltx_trainer.bes.types import GoalTree, Node, VerifierFn


def _temperature(step: int, budget: int, tau_start: float, tau_end: float) -> float:
    if budget <= 1:
        return tau_end
    frac = step / max(budget - 1, 1)
    return tau_start + (tau_end - tau_start) * frac


def boltzmann_sample(
    nodes: list[Node],
    scores: dict[int, float],
    degrees: dict[int, int],
    *,
    tau: float,
    bonus: float,
) -> Node:
    """Single-parent selection (Eq. 3)."""
    weights: list[float] = []
    for i, n in enumerate(nodes):
        s = scores.get(i, 0.0)
        if degrees.get(i, 0) == 0:
            s += bonus
        weights.append(math.exp(s / max(tau, 1e-6)))
    total = sum(weights)
    if total <= 0:
        return random.choice(nodes)
    r = random.random() * total
    acc = 0.0
    for n, w in zip(nodes, weights, strict=True):
        acc += w
        if r <= acc:
            return n
    return nodes[-1]


def boltzmann_pair_sample(
    nodes: list[Node],
    pair_scores_map: dict[tuple[int, int], float],
    *,
    tau: float,
) -> tuple[Node, Node]:
    """Two-parent selection (Eq. 4)."""
    pairs: list[tuple[int, int]] = []
    weights: list[float] = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            pairs.append((i, j))
            w = pair_scores_map.get((i, j), 0.0)
            weights.append(math.exp(w / max(tau, 1e-6)))
    if not pairs:
        a, b = random.sample(nodes, 2) if len(nodes) >= 2 else (nodes[0], nodes[0])
        return a, b
    total = sum(weights)
    r = random.random() * total
    acc = 0.0
    for (i, j), w in zip(pairs, weights, strict=True):
        acc += w
        if r <= acc:
            return nodes[i], nodes[j]
    i, j = pairs[-1]
    return nodes[i], nodes[j]


def sample_operator(probs: dict[str, float], rng: random.Random) -> str:
    ops = list(probs.keys())
    weights = [probs[o] for o in ops]
    return rng.choices(ops, weights=weights, k=1)[0]


def forward_step(
    pool: list[Node],
    eligible: list[int],
    tree: GoalTree,
    verifiers: dict[str, VerifierFn],
    *,
    operator_probs: dict[str, float],
    policy: Callable,
    k_max: int,
    step_idx: int,
    budget: int,
    tau_start: float,
    tau_end: float,
    unexplored_bonus: float,
    parent_degree: dict[int, int],
    rng: random.Random,
    alpha: float,
) -> Node | None:
    """One FORWARDSTEP — Algorithm 2."""
    if not eligible:
        return None

    elig_nodes = [pool[i] for i in eligible]
    local_scores: dict[int, float] = {}
    for li, gi in enumerate(eligible):
        local_scores[li] = score_node(pool[gi].steps, tree, verifiers, alpha=alpha)

    tau = _temperature(step_idx, budget, tau_start, tau_end)
    op = sample_operator(operator_probs, rng)

    if op in ("expand", "delete"):
        degrees_local = {li: parent_degree.get(eligible[li], 0) for li in range(len(eligible))}
        pick = boltzmann_sample(
            elig_nodes,
            local_scores,
            degrees_local,
            tau=tau,
            bonus=unexplored_bonus,
        )
        child = apply_operator(op, pick, None, policy=policy, k_max=k_max, rng=rng)
        if child is not None:
            parent_degree[eligible[elig_nodes.index(pick)]] = (
                parent_degree.get(eligible[elig_nodes.index(pick)], 0) + 1
            )
        return child

    pair_local: dict[tuple[int, int], float] = {}
    for a in range(len(eligible)):
        for b in range(a + 1, len(eligible)):
            pair_local[(a, b)] = score_pair(
                pool[eligible[a]].steps,
                pool[eligible[b]].steps,
                tree,
                verifiers,
                alpha=alpha,
            )
    na, nb = boltzmann_pair_sample(elig_nodes, pair_local, tau=tau)
    child = apply_operator(op, na, nb, policy=policy, k_max=k_max, rng=rng)
    if child is not None:
        for p in (na, nb):
            try:
                gi = eligible[elig_nodes.index(p)]
                parent_degree[gi] = parent_degree.get(gi, 0) + 1
            except ValueError:
                pass
    return child
