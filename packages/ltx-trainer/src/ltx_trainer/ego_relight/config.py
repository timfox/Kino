"""EgoRelight configuration and reference metrics."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EgoRelightConfig:
    paper_arxiv: str = "2605.28401"
    paper_title: str = (
        "EgoRelight: Egocentric Human Capture and Illumination Recovery for "
        "Relightable and Photoreal Avatar Rendering"
    )
    authors: str = "Chen, Zhang, Pandey, Beeler, Habermann, Theobalt"

    uv_size: int = 128
    num_lights: int = 331
    num_specular_rays: int = 32
    blinn_phong_alpha: float = 64.0

    # Depth-conditioned animation thresholds (Eq. 9)
    epsilon_depth_m: float = 0.05
    epsilon_normal: float = 0.5

    # Table 4 reference (Subject #1, Ours)
    ref_psnr: float = 34.81
    ref_ssim: float = 0.9246
    ref_lpips: float = 0.0856
    ref_fid: float = 31.90

    # Runtime breakdown Table 6 (ms per frame, RTX 4090)
    runtime_ms: dict[str, float] = field(
        default_factory=lambda: {
            "ego_pose": 17.62,
            "ik": 67.25,
            "ego_depth": 87.71,
            "animation": 21.50,
            "relight": 100.19,
        }
    )
