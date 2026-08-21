"""D3S Consensus configuration (Luo et al., arXiv:2606.02906)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class D3SOptics:
    """Prototype camera parameters (Sec. 5.1, Table S2)."""

    baseline_mm: float = 3.84
    sensor_distance_mm: float = 12.1
    pixel_pitch_um: float = 2.0
    rho_per_m: float = 83.83
    delta_rho_per_m: float = 0.05
    aperture_radius_mm: float = 1.0
    virtual_baselines_mm: tuple[float, ...] = (0.45, 0.50, 0.55)


@dataclass(frozen=True)
class D3SConsensusParams:
    """Hyperparameters for D3S search and filtering."""

    z_min_m: float = 0.25
    z_max_m: float = 2.0
    z_step_m: float = 0.05
    delta_z_m: float = 0.02
    delta_inv_z_per_m: float = 0.05
    confidence_threshold: float = 0.8
    window_radius: int = 2


@dataclass
class D3SConfig:
    paper_arxiv: str = "2606.02906"
    optics: D3SOptics = field(default_factory=D3SOptics)
    consensus: D3SConsensusParams = field(default_factory=D3SConsensusParams)
    working_range_m: tuple[float, float] = (0.3, 1.64)
    mae_target_cm: float = 1.0
    baselines_compare: tuple[str, ...] = (
        "Intel RealSense D435",
        "Intel RealSense D455",
        "Stereolabs ZED 2",
    )
