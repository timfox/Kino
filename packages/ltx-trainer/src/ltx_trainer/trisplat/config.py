"""TriSplat configuration (Wang et al. arXiv:2605.26115)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MonoNormalBootstrapConfig:
    """Teacher normal blend schedule (paper Eq. 5–6)."""

    takeover_steps: int = 6000
    blend_end_steps: int = 20000


@dataclass
class ProgressiveSharpeningConfig:
    """Opacity exponent and blur multiplier schedules."""

    opacity_e_init: float = 1.0
    opacity_e_final: float = 2.0
    opacity_tau_init: float = 1.0
    opacity_tau_final: float = 5.0
    opacity_schedule_steps: int = 16000
    blur_beta_init: float = 1.0
    blur_beta_final: float = 0.5
    blur_schedule_steps: int = 16000
    alpha_floor: float = 0.02


@dataclass
class MeshExportConfig:
    """Direct triangle export (paper supplement B)."""

    opacity_threshold: float = 0.10
    export_temperature: float = 5.0
    vertex_quantize: float = 1e-5
    subsample_stride: int = 4
    """Keep every Nth pixel triangle to limit mesh size."""


@dataclass
class TriSplatConfig:
    scale_min: float = 0.5
    scale_max: float = 18.0
    template_scale: float = 4.0
    bootstrap: MonoNormalBootstrapConfig = field(default_factory=MonoNormalBootstrapConfig)
    sharpening: ProgressiveSharpeningConfig = field(default_factory=ProgressiveSharpeningConfig)
    export: MeshExportConfig = field(default_factory=MeshExportConfig)
