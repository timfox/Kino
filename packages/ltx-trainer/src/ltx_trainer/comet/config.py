"""COMET: CLAP concept-space dissection via PLS-SVD (arXiv:2605.29628)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CometConfig:
    paper_arxiv: str = "arXiv:2605.29628"
    clap_model: str = "HTSAT-BERT-ZS (WavCaps)"
    embed_dim: int = 1024
    head_size: int = 100
    train_pairs_clotho: int = 19195
    # Table I contribution anchors (Clotho test)
    contrib_direct_pos: float = 0.1937
    contrib_direct100_pos: float = 0.1931
    contrib_cross_pos: float = 0.0326
    contrib_direct_neg: float = 0.0445
    contrib_cross_neg: float = 0.0226
    # Norm energy split (Eq. 5)
    norm_head_text: float = 0.770
    norm_tail_text: float = 0.344
    norm_head_audio: float = 0.776
    norm_tail_audio: float = 0.301
    # PD characterization (Table IV)
    pd_mean_cos_before: float = 0.598
    pd_mean_cos_after: float = 0.995
    pd_head_cos: float = 0.772
    pd_tail_cos: float = 0.095
