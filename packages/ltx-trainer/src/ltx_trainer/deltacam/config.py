"""DeltaCam configuration (arXiv:2605.25266)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IntrinsicRange:
    """Per-parameter physical range for Δ normalization and EXIF mapping."""

    name: str
    m_min: float
    m_max: float
    log_space: bool = True


@dataclass
class DeltaCamConfig:
    """Reference hyperparameters from DeltaCam (Wan-2.1 1.3B, 480×832)."""

    paper_arxiv: str = "arXiv:2605.25266"
    backbone: str = "Wan-2.1-1.3B"
    resolution: tuple[int, int] = (480, 832)

    # Style extraction (Eq. 4)
    lambda_tau: float = 1.0
    lambda_content: float = 1.0
    lambda_style: float = 1.0
    lambda_mi: float = 0.1
    style_dim: int = 256
    random_info_nce_ceiling: float = 4.07

    # wCLIP-5 (Eq. 6)
    wclip_window: int = 2

    # Stage 3: EXIF tokenizer only (paper: <0.1% of full model)
    stage3_trainable_param_fraction: float = 0.001

    # Training curriculum (Sec. 3.5)
    stage1_iterations: int = 60_000
    stage2_iterations: int = 15_000
    stage3_iterations: int = 6_000
    learning_rate: float = 1e-5
    denoising_steps: int = 50

    # Intrinsic groups (Fig. 5)
    optical_params: tuple[str, ...] = ("focal_mm", "aperture_f", "distortion_xi")
    sensory_params: tuple[str, ...] = ("iso", "exposure_ev")
    isp_params: tuple[str, ...] = ("color_temp_k", "shutter_s")

    ranges: dict[str, IntrinsicRange] = field(
        default_factory=lambda: {
            "focal_mm": IntrinsicRange("focal_mm", 10.0, 200.0),
            "aperture_f": IntrinsicRange("aperture_f", 1.4, 22.0),
            "distortion_xi": IntrinsicRange("distortion_xi", 0.0, 1.0, log_space=False),
            "iso": IntrinsicRange("iso", 100.0, 12800.0),
            "exposure_ev": IntrinsicRange("exposure_ev", -3.0, 3.0, log_space=False),
            "color_temp_k": IntrinsicRange("color_temp_k", 2500.0, 10000.0),
            "shutter_s": IntrinsicRange("shutter_s", 1.0 / 4000.0, 1.0 / 15.0),
        }
    )

    def range_for(self, key: str) -> IntrinsicRange:
        if key not in self.ranges:
            raise KeyError(f"Unknown intrinsic {key!r}")
        return self.ranges[key]
