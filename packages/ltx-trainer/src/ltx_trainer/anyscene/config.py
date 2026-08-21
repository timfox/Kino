"""AnyScene driving scene generation (Zhang et al., arXiv:2605.26113)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AnySceneConfig:
    """Defaults from paper Sec. 3–4 and supplement B."""

    # nuCraftv2 / BEV layout
    bev_resolution: int = 256
    bev_channels: int = 15
    occupancy_classes: int = 21
    occupancy_z: int = 25
    voxel_size_m: float = 0.4
    roi_xy_m: float = 51.2
    roi_z_m: float = 5.0
    dataset_hz: int = 12
    sequence_frames_train: int = 4

    # Occupancy VAE (2D BEV-style)
    class_embed_dim: int = 32
    latent_channels: int = 32
    latent_hw: int = 128

    # STOccDiT
    dit_depth: int = 24
    dit_hidden: int = 768
    dit_heads: int = 12
    token_grid: int = 32
    flow_inference_steps: int = 30
    cfg_scale: float = 2.0

    # GGVE (Wan2.1-T2V-14B + VACE ControlNet)
    video_height: int = 480
    video_width: int = 832
    video_frames: int = 49
    control_blocks: int = 8
    plucker_dim: int = 6
    ggve_train_iters: int = 10_000

    # Loss weights (Eq. 2)
    lovasz_weight: float = 1.0
    kl_weight: float = 0.001

    num_surround_views: int = 6
    ggve_modes: tuple[str, ...] = field(
        default_factory=lambda: ("t2v_1", "t2v_2", "t2v_3", "outpaint_1", "outpaint_2")
    )
