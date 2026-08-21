"""FORTE text-to-audio retrieval config (arXiv:2606.05812)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ForteConfig:
    paper_arxiv: str = "arXiv:2606.05812"
    embed_dim: int = 512
    backbones: tuple[str, ...] = ("CLAP", "LAION-CLAP", "Pengi")
    datasets: tuple[str, ...] = ("AudioCaps", "Clotho")
    clotho_clips: int = 4981
    clotho_test_queries: int = 1045
    audiocaps_clips: int = 46000
    audiocaps_test_queries: int = 975
    fol_parser: str = "Flan-T5-XXL (Vossel et al., arXiv:2509.22338)"
    elaboration_llm: str = "Mistral-7B-Instruct-v0.3"
    vocab_predicates: int = 642
    beam_offline: tuple[int, int] = (5, 4)  # B, D
    beam_online: tuple[int, int] = (3, 2)
    feasibility_tau: float = 0.0
    lambda_neg: float = 1.0
    beta_pivot: float = 0.5
    mu_logic: float = 0.1
    alpha_rerank: float = 0.3
    anchor_bank_size: int = 3000
    projection_params_m: float = 1.05
    train_epochs: int = 20
    temperature_init: float = 0.07
    n_pos_elaborations: int = 2
    n_neg_elaborations: int = 3
    metrics_k: tuple[int, ...] = (1, 5, 10, 50)
