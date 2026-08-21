"""Emotion-discriminative zero-shot cross-lingual SER — Mi et al., arXiv:2606.06200."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EdrZsSerConfig:
    paper_arxiv: str = "arXiv:2606.06200"
    title: str = (
        "Learning Emotion-discriminative Representations for "
        "Zero-Shot Cross-lingual Speech Emotion Recognition"
    )
    framework: str = "EDR-ZS-SER"
    affiliations: tuple[str, ...] = ("Nagoya University",)

    # Architecture (§2)
    feature_extractor: str = "wav2vec2-base (language-matched)"
    emotion_classes: tuple[str, ...] = ("happy", "angry", "sad", "neutral")
    hidden_dim: int = 768

    # Loss weights (§2.5, §3.3)
    lambda_crossling: float = 2.5
    alpha_supclr: float = 1.0
    beta_spk_adv: float = 0.3
    contrastive_temperature: float = 0.07

    # Hierarchical batch sampling (§2.2)
    n_lang: int = 3
    n_cls: int = 4
    n_sam: int = 3

    # Languages (§3.1)
    languages: tuple[str, ...] = ("EN", "CN", "DE", "FR", "UR")

    # Table 2 — average over 9 zero-shot settings
    baseline1_avg_uar: float = 59.49
    baseline1_avg_f1: float = 58.24
    baseline2_avg_uar: float = 73.21
    baseline2_avg_f1: float = 72.58
    proposed_avg_uar: float = 82.26
    proposed_avg_f1: float = 81.96
    proposed_wo_supclr_avg_uar: float = 76.86
    proposed_wo_supclr_avg_f1: float = 76.70
    proposed_wo_spkadv_avg_uar: float = 80.11
    proposed_wo_spkadv_avg_f1: float = 80.14
    upper_bound_avg_uar: float = 91.92
    upper_bound_avg_f1: float = 91.41

    # Table 2 — headline EN→DE
    en_de_baseline1_uar: float = 52.23
    en_de_baseline2_uar: float = 88.19
    en_de_proposed_uar: float = 94.64
    en_de_proposed_f1: float = 94.36
    en_de_upper_uar: float = 97.22

    # Ablation deltas vs full proposed (average)
    ablation_supclr_uar_drop: float = 5.40
    ablation_spkadv_uar_drop: float = 2.15
