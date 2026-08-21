"""Configuration for CoRe-KD conversational MER (arXiv:2605.29590)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CoreKDConfig:
    paper_arxiv: str = "arXiv:2605.29590"
    state_dim: int = 256
    hidden_dim: int = 1024
    modalities: tuple[str, ...] = ("text", "audio", "video")
    kd_temperature: float = 2.0
    lambda_kd: float = 1.0
    lambda_state: float = 0.5
    lambda_mstate: float = 0.5
    lambda_nce: float = 1.0
    nce_prob: float = 0.2
    logvar_min: float = -6.0
    logvar_max: float = 2.0
    num_seeds: int = 5
    fold_role: str = "conversational_mer_missing_modality_proxy"
