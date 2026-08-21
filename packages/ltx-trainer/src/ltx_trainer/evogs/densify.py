"""Adaptive densification on evolution-tree frontier (Sec. 3.2)."""

from __future__ import annotations

import numpy as np

from ltx_trainer.evogs.tree import EvolutionTree


def select_split_candidates(
    grad_by_leaf: dict[int, float],
    *,
    threshold: float,
    max_splits: int,
) -> list[int]:
    ranked = sorted(
        ((lid, g) for lid, g in grad_by_leaf.items() if g > threshold),
        key=lambda x: x[1],
        reverse=True,
    )
    return [lid for lid, _ in ranked[:max_splits]]


def split_frontier(
    tree: EvolutionTree,
    leaf_ids: list[int],
    rng: np.random.Generator,
    *,
    dim: int,
) -> int:
    n = 0
    for lid in leaf_ids:
        if lid not in tree.leaves:
            continue
        psi = rng.normal(scale=0.015 + 0.01 * rng.random(), size=dim)
        alpha = rng.uniform(0.6, 1.4, size=tree.cfg.params.alpha_groups)
        tree.split_leaf(lid, psi, alpha)
        n += 1
    return n
