"""CPC3 word-level alignment-aware fusion (arXiv:2605.23604)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Cpc3WlamConfig:
    paper_arxiv: str = "arXiv:2605.23604"
    challenge: str = "3rd Clarity Prediction Challenge (CPC3)"
    backbone_small: str = "openai/whisper-small.en"
    backbone_medium: str = "openai/whisper-medium.en"
    decoder_hidden_dim: int = 256
    severity_embed_dim: int = 128
    dynamic_top_k_heads: int = 10
    local_alignment_mode: str = "character-level dynamic top-10"
    global_pool: str = "masked mean pooling"
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 5
    best_eval_rmse: float = 24.39
    best_eval_corr: float = 0.806
    best_eval_f1: float = 0.778
    best_eval_mcc: float = 0.626
    baseline_rmse: float = 24.92
    baseline_corr: float = 0.795
