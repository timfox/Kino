"""Thaka KSAA-2026 Task 2 Arabic speech diacritization (arXiv:2605.25928)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KsaaDiacConfig:
    paper_arxiv: str = "arXiv:2605.25928"
    task: str = "KSAA-2026 Task 2 — Arabic speech dictation with automatic diacritization"
    team: str = "Thaka, Advanced AI and Information Technology"
    train_samples_raw: int = 2327
    train_samples_filtered: int = 2187
    dev_samples: int = 260
    test_samples: int = 328
    min_diacritic_ratio: float = 0.6
    num_diacritic_classes: int = 15
    catt_layers: int = 6
    hidden_dim: int = 512
    num_heads: int = 16
    whisper_prefix_tokens: int = 150
    whisper_frames_pooled: int = 1500
    total_params_m: int = 39
    trainable_params_m: int = 19
    num_checkpoints: int = 4
    mc_passes_per_checkpoint: int = 50
    mc_dropout_p: float = 0.1
    optuna_trials: int = 30
    # Table 1
    learning_rate: float = 4.1e-6
    rdrop_alpha: float = 2.08
    focal_gamma: float = 0.34
    label_smoothing: float = 0.018
    weight_decay: float = 0.098
    speech_emb_dropout: float = 0.09
    batch_size: int = 16
    epochs: int = 40
    specaug_freq: int = 10
    specaug_time: int = 63
