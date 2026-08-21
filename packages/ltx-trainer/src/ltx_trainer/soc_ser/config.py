"""SOC — second-order correlation SER (Li et al., arXiv:2606.06550)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SocSerConfig:
    paper_arxiv: str = "arXiv:2606.06550"
    title: str = (
        "Geometric Second-Order Feature Correlation Learning for "
        "Self-Supervised Speech Emotion Recognition"
    )
    framework: str = "SOC"
    repo: str = "https://github.com/secret-code-source/SOC"

    # §2 upstream
    ssl_backbones: tuple[str, ...] = ("Wav2Vec2-base", "HuBERT-base", "WavLM-base")
    ssl_dim: int = 768
    default_subspace_d: int = 32

    # Algorithm 1 numerics
    eps_div: float = 1e-6
    eps_id: float = 1e-6

    # §3.2 training
    epochs: int = 100
    peak_lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size_esd: int = 64
    batch_size_ravdess: int = 32
    esd_folds: int = 5
    ravdess_folds: int = 6

    # Table 1 — HuBERT SOC (peak ESD WA)
    hubert_soc_esd_wa: float = 73.50
    hubert_soc_ravdess_wa: float = 69.75
    hubert_gap_esd_wa: float = 71.38
    hubert_soc_wo_lem_esd_wa: float = 72.05

    # Wav2Vec2 SOC vs GAP gains (§3.3)
    w2v_soc_esd_wa: float = 71.86
    w2v_gap_esd_wa: float = 67.18
    w2v_soc_ravdess_wa: float = 58.67
    w2v_gap_ravdess_wa: float = 54.25
    w2v_esd_gain_pct: float = 4.68
    w2v_ravdess_gain_pct: float = 4.42

    # WavLM RAVDESS vs FA (§3.3)
    wavlm_soc_ravdess_wa: float = 68.74
    wavlm_fa_ravdess_wa: float = 66.25
    wavlm_ravdess_gain_vs_fa_pct: float = 2.49

    datasets: tuple[str, ...] = ("ESD", "RAVDESS")
    eval_protocol: str = "EmoBox speaker-independent k-fold"
    baselines: tuple[str, ...] = ("GAP", "ASP", "FA")
