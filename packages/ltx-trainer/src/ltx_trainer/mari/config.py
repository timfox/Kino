"""MARI configuration (arXiv:2605.28722)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MARIConfig:
    arxiv: str = "2605.28722"
    github: str = "https://github.com/V1centNevwake/MARI"
    hidden_dim: int = 4096
    num_adapters: int = 3
    adapter_rank: int = 8
    probe_rank: int = 2
    pca_rank: int = 16
    global_scale_gamma: float = 1.0
    probe_alpha: float = 0.1
    alpha_full: float = 1.0
    alpha_safe: float = 0.0
    target_rejection_rate: float = 0.9
    balance_temperature: float = 1.0
    lambda_off: float = 0.1
    injection_layer: int = 16
    num_post_layers: int = 16
