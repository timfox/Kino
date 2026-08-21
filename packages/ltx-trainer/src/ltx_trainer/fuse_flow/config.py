"""FUSE-Flow — decoupled GMAC + FUSE multi-camera MRR (arXiv:2602.01035)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FuseFlowConfig:
    paper_arxiv: str = "arXiv:2602.01035"
    title: str = (
        "FUSE-Flow: A Decoupled Framework for Calibration and "
        "Stateless Real-Time Multi-View Point Cloud Fusion"
    )
    author: str = "Chentian Sun"
    arxiv_gmac_related: str = "arXiv:2602.01033"

    resolution: tuple[int, int] = (640, 480)
    default_cameras: int = 4
    max_neighbors: int = 4

    # GMAC — Table I ScanNet VGGT (degrees / mm)
    gmac_vggt_rot_scannet: float = 0.63
    gmac_vggt_trans_scannet: float = 3.07
    gmac_baseline_vggt_rot_scannet: float = 1.04
    gmac_baseline_vggt_trans_scannet: float = 4.61

    # GMAC — Table II real-world VGGT (degrees / mm)
    gmac_vggt_rot_real: float = 1.71
    gmac_vggt_trans_real: float = 4.03
    gmac_trigger_acc_pct: float = 96.77

    # FUSE — Table IV static real-world N=8
    fuse_pfr_n8_mm: float = 8.79
    fuse_pncc_n8: float = 0.227
    fuse_or_n8_pct: float = 14.2
    fuse_gpu_mem_n8_gb: float = 2.8
    r3d3_pfr_n8_mm: float = 9.41
    r3d3_gpu_mem_n8_gb: float = 9.1

    # FUSE — Table V dynamic real-world N=8
    fuse_fps_n8: float = 26.3
    r3d3_fps_n8: float = 5.8
    fuse_pfr_dynamic_n8_mm: float = 13.87

    # Pipeline — Table VIII ScanNet 4-cam (GMAC extrinsics → FUSE)
    pipeline_pfr_gmac_mm: float = 10.38
    pipeline_pfr_original_mm: float = 11.73
    pipeline_cd_gmac_mm: float = 8.61
    pipeline_cd_original_mm: float = 9.34

    # Complexity / hardware anchors
    complexity_per_frame: str = "O(NHW)"
    platform_rtx4090: str = "RTX 4090 24GB"

    demo_frames: int = 2
    demo_height: int = 48
    demo_width: int = 64
