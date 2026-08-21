"""Configuration for causal spatio-temporal sound field reconstruction (arXiv:2605.20403)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CausalStSfrConfig:
    """Defaults aligned with paper §VI (image-source / diffuse sims and DTU notes)."""

    paper_id: str = "2605.20403"
    paper_url: str = "https://arxiv.org/abs/2605.20403"
    # Physical
    c_sound: float = 343.0  # m/s
    fs_hz: float = 8000.0
    # Band-limited flat source spectrum (Eq. 14) → κ in Eq. 15
    f1_hz: float = 70.0
    f2_hz: float = 1000.0
    q_source_intensity: float = 1.0
    # Prior geometry (sphere quadrature, Eq. 18)
    sphere_radius_a_m: float = 5.0
    n_quad_points_q: int = 1000
    # Estimator / noise (σ² tuned in paper via LOOM CV)
    sigma2_measurement: float = 1e-3
    # Sample selection (§V, measured DTU experiment)
    relax_eps: float = 1e-9
    prune_rho: float = 1.2
    n_projected_grad_iters: int = 100


@dataclass(frozen=True)
class SimGeometryCard:
    """§VI-A image-source + diffuse free-field simulation card."""

    room_m: tuple[float, float, float] = (3.0, 4.0, 2.5)
    reflection_param: float = 0.5
    mic_array: str = "circular_M8_r0.10_m_center_1.5_1.3_1.2"
    recon_disk: str = "planar_r0.05_m_grid0.01_m_P81"
    discard_conv_samples: int = 800
    t_eval_samples: int = 2000
    nmse_trim_samples: int = 200


@dataclass(frozen=True)
class DtuMeasuredCard:
    """§VI-B/C measured DTU RIR setup (spherical obs, linear validation)."""

    spherical_channels_total: int = 310
    m_obs_draw: int = 50
    p_linear_validation: int = 152
    fs_downsample_hz: float = 8000.0
    causal_horizon_w: int = 20
    full_st_budget_k: int = 1000  # M * W when M=50, W=20
