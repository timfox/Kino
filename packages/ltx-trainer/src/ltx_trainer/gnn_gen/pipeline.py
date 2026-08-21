"""High-level summary tying the three statistical perspectives (Section 5)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ltx_trainer.gnn_gen.config import GNNGenConfig
from ltx_trainer.gnn_gen.convolution import max_degree, symmetric_normalized_adjacency
from ltx_trainer.gnn_gen.csbm import graph_signal_to_noise_ratio, sample_csbm, theorem4_gcn_mu_lower_bound
from ltx_trainer.gnn_gen.graphon import discretization_error_bound, graphon_from_adjacency
from ltx_trainer.gnn_gen.kernels import graph_nngp_gcn, graph_ntk_gcn
from ltx_trainer.gnn_gen.learning_theory import (
    generalization_bound_graph_level,
    rademacher_complexity_gnn_upper_bound,
    transductive_generalization_bound,
    transductive_rademacher_scale,
)
from ltx_trainer.gnn_gen.wl import wl_multiset_signature


@dataclass
class GNNGenSummary:
    """Numeric snapshot for one graph + features."""

    wl_signature_len: int
    rademacher_upper: float
    graphon_discretization_error: float
    nngp_trace: float
    ntk_trace: float
    gsnr: float | None
    theorem4_mu_scale: float | None


def summarize_node_graph(
    x: Tensor,
    adj: Tensor,
    *,
    cfg: GNNGenConfig | None = None,
    csbm_p: float | None = None,
    csbm_q: float | None = None,
) -> GNNGenSummary:
    """Combine learning-theory, kernel, and optional CSBM scales."""
    cfg = cfg or GNNGenConfig()
    n = adj.shape[0]
    sig = wl_multiset_signature(adj, max_iters=cfg.wl_max_iters)
    b = max_degree(adj)
    rad = rademacher_complexity_gnn_upper_bound(
        m=max(n, 1),
        embed_dim=cfg.embed_dim,
        n_layers=cfg.n_gcn_layers,
        max_degree=max(b, 1),
    )
    s = symmetric_normalized_adjacency(adj)
    knngp = graph_nngp_gcn(x, adj, n_layers=cfg.n_gcn_layers, s=s)
    kntk = graph_ntk_gcn(x, adj, n_layers=cfg.n_gcn_layers, s=s)
    gsnr = None
    mu_scale = None
    if csbm_p is not None and csbm_q is not None:
        gsnr = graph_signal_to_noise_ratio(n, csbm_p, csbm_q)
        mu_scale = theorem4_gcn_mu_lower_bound(n, csbm_p, csbm_q)
    return GNNGenSummary(
        wl_signature_len=len(sig),
        rademacher_upper=rad,
        graphon_discretization_error=discretization_error_bound(n),
        nngp_trace=float(torch.trace(knngp).item()),
        ntk_trace=float(torch.trace(kntk).item()),
        gsnr=gsnr,
        theorem4_mu_scale=mu_scale,
    )


def demo_csbm_recovery_check(
    n: int = 200,
    p: float = 0.12,
    q: float = 0.03,
    *,
    seed: int = 0,
) -> dict[str, float | bool]:
    """Sample CSBM and report whether ||mu|| exceeds Theorem 4 GCN scale (heuristic)."""
    mu = torch.ones(8) * 0.5
    x, a, y = sample_csbm(n, p, q, mu, seed=seed)
    norm_mu = float(mu.norm().item())
    thresh = theorem4_gcn_mu_lower_bound(n, p, q)
    return {
        "norm_mu": norm_mu,
        "gcn_mu_threshold_scale": thresh,
        "above_gcn_threshold": norm_mu > thresh,
        "gsnr": graph_signal_to_noise_ratio(n, p, q),
    }
