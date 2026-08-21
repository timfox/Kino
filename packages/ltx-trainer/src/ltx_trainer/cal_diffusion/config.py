"""CAL diffusion co-optimization config (Ye, Khan, Taylor; UC Berkeley)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CALDiffusionConfig:
    """Framework metadata."""

    paper_note: str = "UC Berkeley CAL diffusion + BCLP co-optimization (preprint)"
    packages: tuple[str, ...] = (
        "kernel",
        "tomography",
        "bclp",
        "metrics",
        "identification",
        "simulation",
    )


@dataclass(frozen=True)
class MaterialConfig:
    """PETA / CQ / EDAB / TEMPO photoresin (Methods)."""

    monomer: str = "PETA"
    cq_mm: float = 10.0
    edab_mass_ratio: float = 1.0
    tempo_mm: float = 1.0
    wavelength_nm: float = 459.0
    vial_diameter_mm: float = 6.0
    rotation_deg_s: float = 36.0


@dataclass(frozen=True)
class VoxelConfig:
    """Discretization defaults (47 μm voxels)."""

    voxel_um: float = 47.0
    volume_shape: tuple[int, int, int] = (32, 32, 24)

    @property
    def voxel_m(self) -> float:
        return self.voxel_um * 1e-6


@dataclass(frozen=True)
class DiffusionSweep:
    """Diffusivity grid from kernel identification + correction study."""

    notional_d_m2_s: float = 1.0e-10
    peak_composite_d_m2_s: float = 3.4e-10
    sweep_d_m2_s: tuple[float, ...] = (
        0.5e-10,
        1.0e-10,
        2.0e-10,
        3.0e-10,
        4.0e-10,
        6.0e-10,
    )
    print_time_s: float = 140.0
    uncorrected_print_time_s: float = 135.0
