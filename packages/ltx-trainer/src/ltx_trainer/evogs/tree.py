"""Evolution Tree structure for continuous-layered 3DGS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ltx_trainer.evogs.config import EvoGSConfig, RefinementMode
from ltx_trainer.evogs.gaussians import ghost_mask, random_splats
from ltx_trainer.evogs.refinement import reconstruct_leaf, split_children


@dataclass
class TreeNode:
    node_id: int
    parent_id: int | None
    depth: int
    root_params: np.ndarray
    psi: np.ndarray
    alpha: np.ndarray
    left: int | None = None
    right: int | None = None
    is_leaf: bool = True
    branch_sign: float = 1.0

    def materialized_params(self, nodes: dict[int, TreeNode]) -> np.ndarray:
        """Reconstruct renderable splat at this leaf."""
        if self.parent_id is None:
            return self.root_params.copy()
        chain_psi: list[np.ndarray] = []
        chain_alpha: list[np.ndarray] = []
        signs: list[float] = []
        cur: TreeNode | None = self
        while cur is not None and cur.parent_id is not None:
            parent = nodes[cur.parent_id]
            signs.append(cur.branch_sign)
            chain_psi.append(parent.psi)
            chain_alpha.append(parent.alpha)
            cur = parent
        root = cur.root_params.copy() if cur is not None else self.root_params.copy()
        signs.reverse()
        chain_psi.reverse()
        chain_alpha.reverse()
        return reconstruct_leaf(root, signs, chain_psi, chain_alpha)


@dataclass
class EvolutionTree:
    cfg: EvoGSConfig
    nodes: dict[int, TreeNode] = field(default_factory=dict)
    leaves: set[int] = field(default_factory=set)
    _next_id: int = 0

    def add_root(self, params: np.ndarray) -> int:
        dim = params.shape[0]
        nid = self._next_id
        self._next_id += 1
        self.nodes[nid] = TreeNode(
            node_id=nid,
            parent_id=None,
            depth=0,
            root_params=params.copy(),
            psi=np.zeros(dim),
            alpha=np.ones(min(5, dim)),
            is_leaf=True,
        )
        self.leaves.add(nid)
        return nid

    def split_leaf(
        self,
        leaf_id: int,
        psi: np.ndarray,
        alpha: np.ndarray,
        *,
        mode: RefinementMode | None = None,
    ) -> tuple[int, int]:
        parent = self.nodes[leaf_id]
        mode = mode or self.cfg.params.refinement_mode
        parent.psi = psi.copy()
        parent.alpha = alpha.copy()
        parent.is_leaf = False
        self.leaves.discard(leaf_id)
        c1_params, c2_params = split_children(parent.materialized_params(self.nodes), psi, alpha, mode=mode)
        left_id = self._next_id
        self._next_id += 1
        right_id = self._next_id
        self._next_id += 1
        dim = psi.shape[0]
        self.nodes[left_id] = TreeNode(
            node_id=left_id,
            parent_id=leaf_id,
            depth=parent.depth + 1,
            root_params=parent.root_params.copy(),
            psi=np.zeros(dim),
            alpha=np.ones(min(5, dim)),
            is_leaf=True,
            branch_sign=1.0,
        )
        self.nodes[right_id] = TreeNode(
            node_id=right_id,
            parent_id=leaf_id,
            depth=parent.depth + 1,
            root_params=parent.root_params.copy(),
            psi=np.zeros(dim),
            alpha=np.ones(min(5, dim)),
            is_leaf=True,
            branch_sign=-1.0,
        )
        parent.left = left_id
        parent.right = right_id
        self.leaves.update({left_id, right_id})
        _ = c1_params, c2_params
        return left_id, right_id

    def leaf_params(self) -> np.ndarray:
        rows = [self.nodes[i].materialized_params(self.nodes) for i in sorted(self.leaves)]
        return np.stack(rows, axis=0) if rows else np.zeros((0, self.cfg.params.param_dim))

    def ghost_ratio(self) -> float:
        if not self.leaves:
            return 0.0
        op = self.leaf_params()[:, -1]
        return float(np.mean(ghost_mask(op, self.cfg.params.opacity_ghost_threshold)))

    def storage_bytes(self, level: int) -> int:
        """Cumulative transmitted payload through quality level (internal ψ, α only)."""
        dim = self.cfg.params.param_dim
        alpha_sz = self.cfg.params.alpha_groups
        count = 0
        for node in self.nodes.values():
            if node.depth <= level and node.parent_id is not None and np.any(node.psi != 0):
                count += 1
        per = dim * 4 + alpha_sz * 4
        roots = sum(1 for n in self.nodes.values() if n.parent_id is None)
        return roots * dim * 4 + count * per

    def render_memory_bytes(self) -> int:
        """Materialized leaf set only (ψ, α consumed at reconstruct)."""
        dim = self.cfg.params.param_dim
        return len(self.leaves) * dim * 4

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_nodes": len(self.nodes),
            "num_leaves": len(self.leaves),
            "ghost_ratio": round(self.ghost_ratio(), 4),
            "storage_mb": round(self.storage_bytes(level=3) / 1e6, 2),
            "mem_mb": round(self.render_memory_bytes() / 1e6, 2),
        }


def synthetic_tree(
    n_roots: int = 32,
    splits_per_level: int = 8,
    levels: int = 4,
    *,
    cfg: EvoGSConfig | None = None,
    seed: int = 0,
) -> EvolutionTree:
    cfg = cfg or EvoGSConfig()
    rng = np.random.default_rng(seed)
    tree = EvolutionTree(cfg=cfg)
    dim = cfg.params.param_dim
    for _ in range(n_roots):
        tree.add_root(random_splats(1, dim, rng)[0])
    for level in range(1, levels):
        leaf_ids = list(tree.leaves)
        rng.shuffle(leaf_ids)
        for lid in leaf_ids[:splits_per_level]:
            psi = rng.normal(scale=0.02, size=dim)
            alpha = rng.uniform(0.5, 1.5, size=cfg.params.alpha_groups)
            tree.split_leaf(lid, psi, alpha)
    return tree
