"""SEAM — shortcut-aware scripted vs spontaneous detection (arXiv:2606.06837)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SeamConfig:
    paper_arxiv: str = "arXiv:2606.06837"
    title: str = (
        "SEAM: Shortcut-Aware Real-Time Detection of Scripted vs. Spontaneous "
        "Speech for Interview Guardrails"
    )
    framework: str = "SEAM"
    backbone: str = "DistilHuBERT"
    backbone_params_m: float = 23.49

    # Task (§3.1)
    sample_rate_hz: int = 16000
    window_s: float = 8.0
    chunk_minutes: float = 10.0

    # Preprocessing (§3.2)
    hpf_hz: float = 70.0
    target_lufs: float = -23.0
    peak_limit: float = 0.99

    # Training (§3.3)
    encoder_lr: float = 5e-6
    head_lr: float = 3e-4
    dropout: float = 0.30
    unfreeze_layers: int = 1
    full_epochs: int = 3

    # Noise bank (§3.2)
    noise_bank_hours: float = 14.0
    noise_overlap_min: float = 0.40
    noise_overlap_max: float = 0.70

    # Table 1 — full training mean (3 seeds)
    eval_acc: float = 0.9673
    eval_auc: float = 0.9806
    test_acc: float = 0.9623
    test_auc: float = 0.9766
    ext_acc: float = 0.9517
    ext_auc: float = 0.9713
    ext_auc_std: float = 0.0039

    # Table 5 — INT4 deployment
    int4_vram_mb: float = 41.80
    int4_ext_acc: float = 0.9535
    int4_ext_auc: float = 0.9743
    amp_vram_mb: float = 90.37

    # Table 2 — baseline fixed-budget external AUC
    ablation_baseline_ext_auc: float = 0.8991
    ablation_no_shortcut_ext_auc: float = 0.7324

    # Corpora (§3.2)
    spontaneous_corpora: tuple[str, ...] = ("People's Speech", "PodcastFillers")
    scripted_corpora: tuple[str, ...] = ("LibriSpeech", "Spoken Wikipedia")
