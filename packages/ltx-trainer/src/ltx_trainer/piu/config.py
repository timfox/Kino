"""Configuration for PIU (Bakir et al., arXiv:2605.22311)."""

from __future__ import annotations

from dataclasses import dataclass

BASELINES: tuple[str, ...] = ("siss", "uce", "wid", "arc2face")


@dataclass
class PIUConfig:
    """Defaults from paper Sec. 3–4."""

    embed_dim: int = 512  # ArcFace identity embedding
    clip_dim: int = 768  # padded Arc2Face condition
    preservation_lambda: float = 10.0  # λ in Eq. 4
    negative_guidance_eta: float = 1.5  # η in Eq. 1 (Table 1 uses η=1 for main; default 1.5 ablation)
    anchor_tau: float = 0.2  # cosine similarity target (Sec. 3.3)
    anchor_epsilon: float = 1e-2  # |s_j - τ| tolerance
    dirichlet_alpha: float = 1.0  # forget embedding mix
    surgical_layer_fraction: float = 0.0429  # 4.29% params (Table 6)
    training_steps: int = 400
    batch_size: int = 32
    learning_rate: float = 1e-4
    # Table 1 Arc2Face baseline / PIU
    baseline_forget_ism: float = 0.78
    baseline_retain_ism: float = 0.75
    piu_forget_ism: float = 0.31
    piu_retain_ism: float = 0.72
    piu_srk: float = 88.96
