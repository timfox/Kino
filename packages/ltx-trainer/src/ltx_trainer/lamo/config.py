"""Configuration for LaMo (Jiang et al., arXiv:2605.23878)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LaMoConfig:
    """Defaults from paper Sec. 3 and Appendix B."""

    tau: int = 2
    lambda_drift: float = 0.4
    drift_epsilon: float = 1e-6
    predictor_cosine_weight: float = 0.5
    predictor_aug_prob: float = 0.5
    classifier_free_drop_prob: float = 0.2
    lambda_guide: float = 25.0
    guidance_active_ratio: float = 0.8
    cond_dim: int = 64
    predictor_channels: int = 32
    predictor_blocks: int = 3
