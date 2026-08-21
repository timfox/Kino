"""SSM + MoE architecture configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SSMMoEMode = Literal["moe_mamba", "swimba", "routing_mamba"]


@dataclass
class SSMMoEConfig:
    """Reference config for selective SSM blocks with mixture-of-experts scaling."""

    name: str = "SSM-MoE"
    paper_arxiv: str = "arXiv:2401.04081"
    title: str = "Efficient selective state space models with mixture of experts"
    mode: SSMMoEMode = "moe_mamba"

    d_model: int = 128
    d_state: int = 16
    d_inner: int = 256
    n_layers: int = 4

    num_experts: int = 8
    top_k: int = 2
    expert_capacity_factor: float = 1.25
    load_balance_coeff: float = 0.01

    routing_mamba_shared_router: bool = True
    swimba_single_trajectory: bool = True

    vocab_size: int = 32000
    max_seq_len: int = 2048
