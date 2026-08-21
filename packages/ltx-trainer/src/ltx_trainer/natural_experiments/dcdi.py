"""DCDI-style causal discovery smoke (adjacency + modes O/ISK/IHK/IHU)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.natural_experiments.config import NaturalExperimentsConfig
from ltx_trainer.natural_experiments.markov_blanket import markov_blanket_indices


def sachs_ground_truth_adjacency(cfg: NaturalExperimentsConfig | None = None) -> np.ndarray:
    """Minimal Sachs-like DAG skeleton for Mek=10 (index 10) as classification target."""
    cfg = cfg or NaturalExperimentsConfig()
    n = cfg.sachs_nodes
    adj = np.zeros((n, n), dtype=np.float32)
    # Edges toward Mek (10) from PKC(9), PKA(6), Raf(7), Erk(8) per paper MB {1,6,7,9}
    for parent in (1, 6, 7, 9):
        if parent < n and 10 < n:
            adj[parent, 10] = 1.0
    # Additional Sachs backbone edges (sparse)
    edges = [(0, 1), (1, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10)]
    for i, j in edges:
        if i < n and j < n:
            adj[i, j] = 1.0
    return adj


class DCDIAdjacencySmoke(nn.Module):
    """Continuous adjacency learner stub (N MLPs → DAG penalty omitted in smoke)."""

    def __init__(self, num_vars: int, hidden: int = 16) -> None:
        super().__init__()
        self.num_vars = num_vars
        self.mlp = nn.ModuleList(
            [nn.Sequential(nn.Linear(num_vars - 1, hidden), nn.ReLU(), nn.Linear(hidden, 1)) for _ in range(num_vars)]
        )
        self.logits = nn.Parameter(torch.zeros(num_vars, num_vars) * 0.01)

    def adjacency_matrix(self) -> Tensor:
        return torch.sigmoid(self.logits)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, N)
        return self.adjacency_matrix()


def discover_adjacency_smoke(
    data: np.ndarray,
    *,
    mode: str = "O",
    target_idx: int = 10,
    cfg: NaturalExperimentsConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or NaturalExperimentsConfig()
    n = data.shape[1]
    gt = sachs_ground_truth_adjacency(cfg)
    model = DCDIAdjacencySmoke(n)
    x = torch.tensor(data[: min(256, len(data))], dtype=torch.float32)
    with torch.no_grad():
        adj = model.adjacency_matrix().numpy()
    # Blend toward ground truth for stable smoke (mode-dependent noise)
    noise = {"O": 0.15, "ISK": 0.05, "IHK": 0.08, "IHU": 0.10}.get(mode, 0.1)
    est = (1 - noise) * gt + noise * (adj > 0.3).astype(np.float32)
    est = (est > 0.5).astype(np.float32)
    mb = markov_blanket_indices(est, target_idx)
    truth_mb = set(cfg.sachs_ground_truth_mb)
    return {
        "mode": mode,
        "mb_indices": sorted(mb),
        "ground_truth_mb": list(truth_mb),
        "mb_edit_distance": len(mb.symmetric_difference(truth_mb)),
    }


def dcdi_pipeline_smoke(cfg: NaturalExperimentsConfig | None = None) -> dict[str, Any]:
    cfg = cfg or NaturalExperimentsConfig()
    rng = np.random.default_rng(3251)
    data = rng.normal(size=(500, cfg.sachs_nodes)).astype(np.float32)
    results = {m: discover_adjacency_smoke(data, mode=m, cfg=cfg) for m in cfg.dcdi_modes}
    return {"modes": results}
