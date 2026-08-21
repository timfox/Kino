"""CAST configuration (Lu et al., arXiv:2605.16919)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_URL = "https://arxiv.org/abs/2605.16919"
PAPER_TITLE = (
    "CAST: Causal Anchored Simplex Transport for Distribution-Valued Time Series"
)

# Paper benchmark aggregate ranks (Table 2)
KL_AVG_RANK = 1.27
ROLLOUT_JSD_AVG_RANK = 1.91
MEAN_RANK = 1.59
TOP1_KL_WINS = 8
TOP1_RO_WINS = 8
NUM_BENCHMARK_SECTIONS = 11

# CAST hyperparameters (Appendix D.4)
RETRIEVAL_HEADS = 2
RETRIEVAL_DIM = 64
LAMBDA_MIN = 0.05
LAMBDA_MAX = 0.95
TRANSPORT_RADIUS = 1
RHO_MAX = 0.20
DELTA_MU = 0.25
DELTA_SIGMA = 0.10
LAMBDA_OP = 5e-4
SMOOTHNESS_WEIGHT = 1e-4
METRIC_EPS = 1e-8


@dataclass
class CASTConfig:
    support_dim: int = 32
    hidden_dim: int = 64  # paper: 256; stub uses smaller for CPU tests
    num_layers: int = 2  # paper: 6
    num_heads: int = 4
    retrieval_heads: int = RETRIEVAL_HEADS
    retrieval_dim: int = RETRIEVAL_DIM
    dropout: float = 0.1
    ordered_support: bool = True
    lambda_min: float = LAMBDA_MIN
    lambda_max: float = LAMBDA_MAX
    rho_max: float = RHO_MAX
    delta_mu: float = DELTA_MU
    delta_sigma: float = DELTA_SIGMA
    lambda_op: float = LAMBDA_OP
    max_seq_len: int = 128
