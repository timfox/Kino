"""Stable hybrid cross-attention fusion for AVER (arXiv:2606.03747)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HybridAverConfig:
    paper_arxiv: str = "arXiv:2606.03747"
    title: str = "Stable Hybrid Cross-Attention Fusion for Audio-Visual Event Recognition"
    dataset: str = "AVE"
    video_backbone: str = "MCG-NJU/videomae-base"
    audio_backbone: str = "MIT/ast-finetuned-audioset-10-10-0.4593"

    num_segments: int = 10
    embed_dim: int = 512
    num_heads: int = 8
    num_fusion_layers: int = 2
    classifier_dropout: float = 0.5
    num_classes: int = 28

    # Table I — Proposed Hybrid Fusion (5 seeds, mean ± std)
    best_val_acc_mean: float = 0.8948
    best_val_acc_std: float = 0.0084
    best_val_acc_peak: float = 0.9174  # abstract headline
    test_acc_mean: float = 0.8385
    test_acc_std: float = 0.0140
    balanced_acc_mean: float = 0.8277
    weighted_f1_mean: float = 0.8368

    # Table II
    params_m_hybrid: float = 6.856
    train_hours_hybrid: float = 0.3835
    split_train: int = 1891
    split_val: int = 230
    split_test: int = 234

    demo_batch: int = 2
    demo_time_steps: int = 10
