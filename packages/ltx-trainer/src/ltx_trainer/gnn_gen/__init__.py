"""GNN generalisation: three statistical perspectives (Ayday, Sabanayagam, Ghoshdastidar; arXiv:2605.25452).

Reference utilities for learning-theoretic bounds, 1-WL expressivity, Graph NNGP/NTK, graphon limits,
and CSBM — not a full GNN training stack.
"""

from ltx_trainer.gnn_gen.config import GNNGenConfig
from ltx_trainer.gnn_gen.csbm import (
    features_linearly_separable_threshold,
    graph_signal_to_noise_ratio,
    sample_csbm,
    theorem4_gat_mu_lower_bound,
    theorem4_gcn_mu_lower_bound,
)
from ltx_trainer.gnn_gen.graphon import (
    discretization_error_bound,
    graphon_convolution_apply,
    graphon_from_adjacency,
    graphon_gcn_layer,
)
from ltx_trainer.gnn_gen.kernels import graph_nngp_gcn, graph_ntk_gcn
from ltx_trainer.gnn_gen.learning_theory import (
    empirical_rademacher_estimate,
    generalization_bound_graph_level,
    rademacher_complexity_gnn_upper_bound,
    transductive_generalization_bound,
    transductive_rademacher_scale,
)
from ltx_trainer.gnn_gen.linear_gcn import linear_gcn_forward, linear_gcn_predict, oversmoothing_rank_one_limit
from ltx_trainer.gnn_gen.pipeline import GNNGenSummary, demo_csbm_recovery_check, summarize_node_graph
from ltx_trainer.gnn_gen.wl import (
    cycle_graph,
    path_graph,
    wl_colors,
    wl_multiset_signature,
    wl_same_under_refinement,
)

__all__ = [
    "GNNGenConfig",
    "GNNGenSummary",
    "cycle_graph",
    "demo_csbm_recovery_check",
    "discretization_error_bound",
    "empirical_rademacher_estimate",
    "features_linearly_separable_threshold",
    "generalization_bound_graph_level",
    "graph_nngp_gcn",
    "graph_ntk_gcn",
    "graph_signal_to_noise_ratio",
    "graphon_convolution_apply",
    "graphon_from_adjacency",
    "graphon_gcn_layer",
    "linear_gcn_forward",
    "linear_gcn_predict",
    "oversmoothing_rank_one_limit",
    "path_graph",
    "rademacher_complexity_gnn_upper_bound",
    "sample_csbm",
    "summarize_node_graph",
    "theorem4_gat_mu_lower_bound",
    "theorem4_gcn_mu_lower_bound",
    "transductive_generalization_bound",
    "transductive_rademacher_scale",
    "wl_colors",
    "wl_multiset_signature",
    "wl_same_under_refinement",
]
