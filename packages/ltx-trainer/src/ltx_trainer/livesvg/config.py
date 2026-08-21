"""LiveSVG configuration (Levy et al., arXiv:2605.30174)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LiveSVGConfig:
    paper_arxiv: str = "arXiv:2605.30174"
    project_page: str = "https://levymsn.github.io/LiveSVG"
    num_keyframes: int = 15
    opt_resolution: int = 256
    export_pixels: int = 720
    opt_iterations: int = 2000
    candidate_videos_min: int = 10
    candidate_videos_max: int = 20
    lambda_mse: float = 1000.0
    lambda_spatial: float = 0.5
    lambda_g1: float = 10.0
    lambda_sdf: float = 1.0
    spatial_sigma_frac: float = 0.01
    gaussian_kernel_size: int = 5
    gaussian_sigma: float = 1.0
    g1_angle_threshold_deg: float = 10.0
    tap_confidence: float = 0.9
    progressive_interval_iters: int = 100
    lr_homography: float = 1e-3
    lr_path_offsets: float = 1e-1
    sdf_margin_px: float = 1.0
    reference_generators: tuple[str, ...] = ("veo3.1", "ltx2.3", "wan2.2")
    default_reference_generator: str = "veo3.1"
    semantic_grouper: str = "gemini-3.1-pro"
    video_filter_model: str = "gemini-3.1-pro"
    renderer: str = "diffvg"
    tracking_backend: str = "tapnext"
    layer_order_backend: str = "sam2"
    foreground_mask_backend: str = "rmbg-1.4"
    aniclipart_examples: int = 43
    challengesvg_examples: int = 35


BASELINES: tuple[str, ...] = (
    "Vector Prism",
    "LiveSketch",
    "AniClipart",
    "FlexiClip",
    "LINR-Bridge",
)
