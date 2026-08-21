"""Progressive multi-level training (Sec. 3.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ltx_trainer.evogs.config import EvoGSConfig
from ltx_trainer.evogs.densify import select_split_candidates, split_frontier
from ltx_trainer.evogs.tree import EvolutionTree, synthetic_tree


@dataclass
class LevelState:
    level: int
    downsample: int
    leaf_count: int
    ghost_ratio: float
    storage_mb: float
    mem_mb: float


def image_pyramid_factors(cfg: EvoGSConfig) -> tuple[int, ...]:
    return cfg.params.downsample_factors[: cfg.params.num_levels]


def train_level_stub(
    tree: EvolutionTree,
    level: int,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Simulate one progressive stage: unfrozen frontier ψ, α; gradient-driven splits.

    Returns metrics for the stage without CUDA rasterization.
    """
    rng = np.random.default_rng(seed + level * 17)
    dim = tree.cfg.params.param_dim
    grads = {lid: float(rng.exponential(1.0)) for lid in tree.leaves}
    candidates = select_split_candidates(
        grads,
        threshold=tree.cfg.params.grad_split_threshold * 1e4,
        max_splits=max(1, len(tree.leaves) // 4),
    )
    n_split = split_frontier(tree, candidates, rng, dim=dim)
    return {
        "level": level,
        "candidates": len(candidates),
        "splits": n_split,
        "leaf_count": len(tree.leaves),
        "ghost_ratio": tree.ghost_ratio(),
    }


def progressive_train_demo(
    *,
    cfg: EvoGSConfig | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    cfg = cfg or EvoGSConfig()
    tree = synthetic_tree(n_roots=24, splits_per_level=6, levels=1, cfg=cfg, seed=seed)
    stages: list[dict[str, Any]] = []
    for level in range(cfg.params.num_levels):
        stages.append(train_level_stub(tree, level, seed=seed))
    levels = [
        LevelState(
            level=i,
            downsample=image_pyramid_factors(cfg)[i],
            leaf_count=len(tree.leaves),
            ghost_ratio=tree.ghost_ratio(),
            storage_mb=tree.storage_bytes(i) / 1e6,
            mem_mb=tree.render_memory_bytes() / 1e6,
        )
        for i in range(cfg.params.num_levels)
    ]
    return {
        "stages": stages,
        "levels": [level.__dict__ for level in levels],
        "tree": tree.to_dict(),
        "pyramid": list(image_pyramid_factors(cfg)),
    }
