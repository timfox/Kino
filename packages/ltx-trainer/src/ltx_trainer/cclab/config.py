"""Configuration for CCLab adversarial CC testing (Chen et al., arXiv:2605.21915)."""

from __future__ import annotations

from dataclasses import dataclass

LEARNING_CC: tuple[str, ...] = ("orca", "canopy")
NON_LEARNING_CC: tuple[str, ...] = ("bbr", "vegas", "cubic", "illinois", "tcp_lp")
FEATURE_LEVEL_CC: tuple[str, ...] = ("orca", "canopy", "bbr", "vegas")
ENV_LEVEL_CC: tuple[str, ...] = ("orca", "canopy", "cubic", "vegas", "illinois", "tcp_lp")


@dataclass
class CCLabConfig:
    """Defaults from paper Sec. III–VII."""

    # Emulator (Sec. III)
    link_delay_ms: float = 10.0
    queue_bdp_multiplier: float = 2.0
    # CC reward (Eq. 1–2)
    loss_penalty_lambda: float = 1.0
    delay_margin_gamma: float = 1.0
    # Adversarial delay window (Eq. 5)
    delay_history_h: int = 5
    delay_instant_k: int = 1
    delay_penalty_alpha: float = 1.0
    # Feature perturbation
    perturb_small: float = 0.05  # ±5%
    perturb_large: float = 0.50  # ±50%
    # Environment bandwidth (Sec. V)
    bw_min_mbps: float = 1.0
    bw_max_mbps: float = 96.0
    smoothness_budget_delta_mbps: float = 48.0
    smoothness_window_k: int = 4
    # Adversarial training (Sec. VII)
    adv_train_mixing_ratio: float = 0.2
    # Reference degradation under ±50% feature attack (abstract)
    orca_feature_degradation_pct: float = 8.0
    canopy_feature_degradation_pct: float = 3.0
    bbr_feature_degradation_pct: float = 15.0
    vegas_feature_degradation_pct: float = 10.0
