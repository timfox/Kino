"""Framework card, benchmark tables, evaluation demo."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.fuse_flow.config import FuseFlowConfig
from ltx_trainer.fuse_flow.fuse import fuse_views_stub
from ltx_trainer.fuse_flow.gmac import gmac_refine_stub
from ltx_trainer.fuse_flow.ltx_plan import ltx_integration_plan


def framework_card(cfg: FuseFlowConfig | None = None) -> dict[str, Any]:
    cfg = cfg or FuseFlowConfig()
    return {
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "author": cfg.author,
        "modules": {
            "GMAC": "metric-scale lifting via RC + cycle consistency",
            "FUSE": "MCM + CVGC + ASH/RPS stateless fusion",
        },
        "pipeline": "RGB-D streams → GMAC extrinsics → FUSE point cloud",
        "complexity": cfg.complexity_per_frame,
        "resolution": list(cfg.resolution),
        "decoupled": True,
        "stateless": True,
        "backbones": ["VGGSfM", "MapAnything", "VGGT"],
    }


def table1_gmac_scannet() -> list[dict[str, Any]]:
    cfg = FuseFlowConfig()
    return [
        {
            "backbone": "VGGT",
            "method": "Original",
            "rot_error_deg": cfg.gmac_baseline_vggt_rot_scannet,
            "trans_error_mm": cfg.gmac_baseline_vggt_trans_scannet,
        },
        {
            "backbone": "VGGT",
            "method": "GMAC",
            "rot_error_deg": cfg.gmac_vggt_rot_scannet,
            "trans_error_mm": cfg.gmac_vggt_trans_scannet,
        },
    ]


def table2_gmac_realworld() -> list[dict[str, Any]]:
    cfg = FuseFlowConfig()
    return [
        {
            "backbone": "VGGT",
            "method": "GMAC",
            "rot_error_deg": cfg.gmac_vggt_rot_real,
            "trans_error_mm": cfg.gmac_vggt_trans_real,
        }
    ]


def table4_fuse_static_realworld() -> list[dict[str, Any]]:
    cfg = FuseFlowConfig()
    return [
        {
            "method": "R3D3",
            "cameras": 8,
            "pfr_mm": cfg.r3d3_pfr_n8_mm,
            "gpu_mem_gb": cfg.r3d3_gpu_mem_n8_gb,
        },
        {
            "method": "FUSE",
            "cameras": 8,
            "pfr_mm": cfg.fuse_pfr_n8_mm,
            "pncc": cfg.fuse_pncc_n8,
            "or_pct": cfg.fuse_or_n8_pct,
            "gpu_mem_gb": cfg.fuse_gpu_mem_n8_gb,
        },
    ]


def table5_fuse_dynamic_realworld() -> list[dict[str, Any]]:
    cfg = FuseFlowConfig()
    return [
        {"method": "R3D3", "cameras": 8, "fps": cfg.r3d3_fps_n8, "pfr_mm": 15.82},
        {
            "method": "FUSE",
            "cameras": 8,
            "fps": cfg.fuse_fps_n8,
            "pfr_mm": cfg.fuse_pfr_dynamic_n8_mm,
        },
    ]


def table8_pipeline_extrinsics() -> list[dict[str, Any]]:
    cfg = FuseFlowConfig()
    return [
        {
            "dataset": "ScanNet",
            "extrinsic": "Original",
            "pfr_mm": cfg.pipeline_pfr_original_mm,
            "cd_mm": cfg.pipeline_cd_original_mm,
        },
        {
            "dataset": "ScanNet",
            "extrinsic": "GMAC",
            "pfr_mm": cfg.pipeline_pfr_gmac_mm,
            "cd_mm": cfg.pipeline_cd_gmac_mm,
        },
    ]


def forward_smoke(cfg: FuseFlowConfig | None = None) -> dict[str, Any]:
    cfg = cfg or FuseFlowConfig()
    g = torch.Generator().manual_seed(0)
    h, w = cfg.demo_height, cfg.demo_width
    depths = [
        torch.rand(h, w, generator=g) * 1.5 + 0.5,
        torch.rand(h, w, generator=g) * 1.5 + 0.5,
        torch.rand(h, w, generator=g) * 1.5 + 0.5,
    ]
    return {
        "gmac": gmac_refine_stub(n_cameras=3, seed=0),
        "fuse": fuse_views_stub(depths, cfg=cfg),
    }


def evaluation_demo(cfg: FuseFlowConfig | None = None) -> dict[str, Any]:
    cfg = cfg or FuseFlowConfig()
    return {
        "framework": framework_card(cfg),
        "table1_gmac_scannet": table1_gmac_scannet(),
        "table2_gmac_realworld": table2_gmac_realworld(),
        "table4_fuse_static": table4_fuse_static_realworld(),
        "table5_fuse_dynamic": table5_fuse_dynamic_realworld(),
        "table8_pipeline": table8_pipeline_extrinsics(),
        "forward": forward_smoke(cfg),
        "ltx_evolution": ltx_integration_plan(cfg),
    }
