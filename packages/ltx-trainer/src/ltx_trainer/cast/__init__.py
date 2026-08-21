"""CAST: Causal Anchored Simplex Transport (Lu et al., arXiv:2605.16919)."""

from ltx_trainer.cast.config import CASTConfig, PAPER_TITLE, PAPER_URL
from ltx_trainer.cast.encoder import CausalTransformerEncoder, SimplexEmbedding
from ltx_trainer.cast.losses import cast_loss, operator_regularizer, step_kl_loss
from ltx_trainer.cast.metrics import (
    table1_benchmarks,
    table2_average_ranks,
    table3_cast_wins,
    table4_ablation,
    table5_aliasing_experiment,
)
from ltx_trainer.cast.model import CAST
from ltx_trainer.cast.pipeline import aliasing_js_lower_bound, count_parameters, paper_report, persistence_baseline, train_step
from ltx_trainer.cast.retrieval import CausalRetrieval
from ltx_trainer.cast.simplex import (
    jensen_shannon,
    kl_divergence,
    l1_distance,
    normalize_simplex,
    support_mean,
    w1_ordered,
    weighted_js,
)
from ltx_trainer.cast.synthetic import aliasing_setup, compositional_sequence, dataset_summary, queue_occupancy_sequence
from ltx_trainer.cast.transport import TransportHead, apply_local_transport

__all__ = [
    "CAST",
    "CASTConfig",
    "CausalRetrieval",
    "CausalTransformerEncoder",
    "PAPER_TITLE",
    "PAPER_URL",
    "SimplexEmbedding",
    "TransportHead",
    "aliasing_js_lower_bound",
    "aliasing_setup",
    "apply_local_transport",
    "cast_loss",
    "compositional_sequence",
    "count_parameters",
    "dataset_summary",
    "jensen_shannon",
    "kl_divergence",
    "l1_distance",
    "normalize_simplex",
    "operator_regularizer",
    "paper_report",
    "persistence_baseline",
    "queue_occupancy_sequence",
    "step_kl_loss",
    "support_mean",
    "table1_benchmarks",
    "table2_average_ranks",
    "table3_cast_wins",
    "table4_ablation",
    "table5_aliasing_experiment",
    "train_step",
    "w1_ordered",
    "weighted_js",
]
