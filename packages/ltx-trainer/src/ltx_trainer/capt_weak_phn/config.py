"""Weakly-supervised phoneme-level pronunciation scoring (arXiv:2605.23593)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CaptWeakPhnConfig:
    paper_arxiv: str = "arXiv:2605.23593"
    dataset: str = "Speechocean762"
    task: str = "Phoneme-level pronunciation (accuracy) scoring under weak supervision"
    inventory_k: int = 42
    gop_feature_dim: int = 84  # 2K
    transformer_hidden: int = 24
    pooling_strategies: tuple[str, ...] = ("BASE", "MEAN", "ATTN")
    attention_pooling: str = "attention-weighted average of phoneme scores within unit"
    weak_supervision_levels: tuple[str, ...] = ("UWP", "P", "W", "UW", "U")
    stage1_utterances_n: int = 2500
    stage2_budgets: tuple[int, ...] = (100, 250, 500, 1000, 2500)

    # Fixed paper excerpts (development/test highlights).
    gop_phoneme_pcc: float = 0.34
    table1_u_attn_phn_pcc: float = 0.46
    table1_u_attn_phn_mse: float = 0.23
    fig3_test_gop_pcc: float = 0.34
