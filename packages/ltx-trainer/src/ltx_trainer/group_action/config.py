"""Configuration for group-action world models (Wang et al., arXiv:2605.24578)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GroupActionConfig:
    """GA regularization + GAC/GAR evaluation defaults (Sec. 3–4, Appendix B/C)."""

    # Ego-motion increment (dx, dy, dtheta) per step — operational SE(2) proxy (Appendix A.4)
    action_dim: int = 3
    rotation_weight: float = 1.0  # alpha in Eq. (8)

    # Latent GA loss weights (Eq. 12)
    lambda_id: float = 1.0
    lambda_inv: float = 1.0
    lambda_comp: float = 1.0
    lambda_ga: float = 0.5  # weight on L_GA in L = L_diff + lambda_ga * L_GA (Eq. 13)

    # Training rollout (Appendix B)
    max_rollout_horizon: int = 4
    dirichlet_alpha: float = 1.0  # composition redistribution (Eq. 17)

    # GAC probe grids (Sec. 4.1, Appendix C.6)
    identity_segment_counts: tuple[int, ...] = (1, 3, 5)
    identity_segment_lengths: tuple[int, ...] = (1, 3, 5)
    composition_window_lengths: tuple[int, ...] = (2, 4, 6)

    # GAR (Eq. 21)
    gar_num_rollouts: int = 5
