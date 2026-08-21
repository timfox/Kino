"""MeshToken render-free human motion control (arXiv:2606.02000)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class MeshTokenConfig:
    paper_arxiv: str = "arXiv:2606.02000"
    title: str = (
        "Towards 3D-Aware Video Diffusion Models: Render-Free Human Motion "
        "Control with Mesh Tokenization"
    )
    project_url: str = "https://jingyunliang.github.io/MeshToken"
    backbone: str = "Wan-2.1 I2V (frozen)"
    params_b: float = 20.33

    # SMPL / mesh (§3.2)
    smpl_vertices: int = 6890
    smpl_faces: int = 13776
    pose_tokens: int = 54
    pose_token_dim: int = 9
    dit_hidden: int = 5120
    temporal_downsample: int = 4
    train_frames: int = 97
    latent_frames: int = 25

    # Mesh VQ (VQ-HPS [10])
    codebook_size: int = 8192
    mesh_recon_error_mm: Tuple[float, float] = (8.0, 7.1)  # Trajectory100, RealisDance-Val

    # Inference (§4.1)
    cfg_scale: float = 5.0
    ddim_steps: int = 40

    # Table 1 Trajectory100
    traj100_translation_error_m: float = 1.697
    traj100_rotation_error_deg: float = 0.173
    traj100_psnr: float = 16.78
    traj100_fvd: float = 695.62

    # Table 2 RealisDance-Val highlights (best or ours)
    realis_i2v_bg: float = 97.18
    realis_bg_consist: float = 96.23
    realis_aesthetic: float = 58.14
