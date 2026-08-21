"""Bridge: graph-conditioned LoRA A-matrix for LTX DiT PEFT layers."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from ltx_trainer.ilora.config import ILORAConfig
from ltx_trainer.ilora.gnn import GraphEncoder
from ltx_trainer.ilora.graph import PoissonLaplaceGraphBranch
from ltx_trainer.ilora.lora import GraphHyperLoRA


class ILORALTXAdapterBridge(nn.Module):
    """Sample-conditioned LoRA A for a single DiT linear layer (conceptual iLoRA → LTX)."""

    def __init__(self, cfg: ILORAConfig | None = None, *, d_in: int | None = None, d_out: int | None = None) -> None:
        super().__init__()
        cfg = cfg or ILORAConfig()
        d_in = d_in or cfg.ltx_target_hidden
        d_out = d_out or cfg.ltx_target_hidden
        self.node_proj = nn.Linear(1, cfg.graph_dim)
        self.graph_branch = PoissonLaplaceGraphBranch(cfg.graph_dim)
        self.gnn = GraphEncoder(cfg.graph_dim, cfg.graph_dim)
        self.hyper_lora = GraphHyperLoRA(cfg.graph_dim, d_in, d_out, cfg.ltx_lora_rank, alpha=cfg.lora_alpha)
        self.d_out = d_out

    def forward(self, node_features: Tensor) -> dict[str, Tensor | float]:
        """
        node_features: [B, K] abundances or latent entity scores.
        Returns LoRA A [B, r, d_in], adjacency, KL regularizers.
        """
        h = self.node_proj(node_features.unsqueeze(-1))
        adj, pois_kl, lap_kl = self.graph_branch(h)
        h_graph = self.gnn(h, adj)
        a = self.hyper_lora.generate_a(h_graph)
        return {
            "lora_a": a,
            "adjacency": adj,
            "pois_kl": pois_kl,
            "lap_kl": lap_kl,
            "h_graph": h_graph,
        }


def ltx_peft_integration_notes(cfg: ILORAConfig | None = None) -> dict[str, Any]:
    """How iLoRA-style hypernetwork maps onto ltx-trainer PEFT."""
    cfg = cfg or ILORAConfig()
    return {
        "target": "Replace static LoRA A with GraphHyperLoRA.generate_a(h_graph) per batch/sample",
        "freeze": ["DiT backbone W0", "LLM backbone in microbiome setting"],
        "train": ["graph branch", "GNN", "hypernetwork proj_a", "static lora_b"],
        "inference_merge": "Optional: average A over MC graph samples S before merge into W0+sBA",
        "rank": cfg.ltx_lora_rank,
        "reference_loader": "kino/packages/ltx-trainer/scripts/inference.py load_lora_weights",
    }
