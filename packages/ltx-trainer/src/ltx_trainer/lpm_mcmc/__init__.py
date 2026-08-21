"""Fast MCMC for latent position models (Li & Smith, arXiv:2605.30134)."""

from ltx_trainer.lpm_mcmc.complexity import (
    fast_sweep_cost,
    faster_sweep_cost,
    naive_sweep_cost,
    rmf24_sweep_cost,
)
from ltx_trainer.lpm_mcmc.config import LPMMCMCConfig
from ltx_trainer.lpm_mcmc.indexing import alpha_size, multi_indices, projected_indices
from ltx_trainer.lpm_mcmc.link import exact_log_likelihood, gaussian_edge_prob
from ltx_trainer.lpm_mcmc.mcmc import MCMCState, propose_embedding_step, run_embedding_sweep
from ltx_trainer.lpm_mcmc.metrics import complexity_comparison, empirical_mse_posterior_mean
from ltx_trainer.lpm_mcmc.moments import MomentStore
from ltx_trainer.lpm_mcmc.partition import block_centers, block_partition, is_b_good
from ltx_trainer.lpm_mcmc.pipeline import (
    benchmark_manifest,
    evaluation_demo,
    framework_card,
    paper_limitations,
    training_step_demo,
)
from ltx_trainer.lpm_mcmc.taylor import approximate_log_likelihood, taylor_error_bound, tv_error_bound

__all__ = [
    "LPMMCMCConfig",
    "MCMCState",
    "MomentStore",
    "alpha_size",
    "approximate_log_likelihood",
    "benchmark_manifest",
    "block_centers",
    "block_partition",
    "complexity_comparison",
    "empirical_mse_posterior_mean",
    "evaluation_demo",
    "exact_log_likelihood",
    "fast_sweep_cost",
    "faster_sweep_cost",
    "framework_card",
    "gaussian_edge_prob",
    "is_b_good",
    "multi_indices",
    "naive_sweep_cost",
    "paper_limitations",
    "projected_indices",
    "propose_embedding_step",
    "rmf24_sweep_cost",
    "run_embedding_sweep",
    "taylor_error_bound",
    "training_step_demo",
    "tv_error_bound",
]
