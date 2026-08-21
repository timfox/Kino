"""MeshToken framework card, tables, evaluation demo."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.mesh_token.config import MeshTokenConfig
from ltx_trainer.mesh_token.dit_motion import dit_motion_smoke
from ltx_trainer.mesh_token.smpl import smpl_motion_stub
from ltx_trainer.mesh_token.tokenization import MotionTokenizer


def framework_card(cfg: MeshTokenConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MeshTokenConfig()
    return {
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "url": cfg.project_url,
        "backbone": cfg.backbone,
        "params_b": cfg.params_b,
        "representation": {
            "motion": "SMPL mesh → VQ pose tokens + trajectory MLP",
            "render_free": True,
            "injection": "per-latent-frame motion cross-attention (frozen Wan-2.1 DiT)",
        },
        "tokenization": {
            "pose_tokens": f"{cfg.pose_tokens}×{cfg.pose_token_dim}",
            "dit_hidden": cfg.dit_hidden,
            "latent_frames": cfg.latent_frames,
        },
    }


def table1_trajectory100(cfg: MeshTokenConfig | None = None) -> list[dict[str, Any]]:
    cfg = cfg or MeshTokenConfig()
    return [
        {"method": "Wan-2.1-I2V", "translation_error_m": 10.349, "rotation_error_deg": 0.418, "psnr": 14.96, "fvd": 1421.87},
        {"method": "Tora", "translation_error_m": 5.667, "rotation_error_deg": 0.355, "psnr": 16.56, "fvd": 957.81},
        {"method": "RealisDance-DiT", "translation_error_m": 1.706, "rotation_error_deg": 0.167, "psnr": 16.17, "fvd": 758.08},
        {"method": "RealisMotion*", "translation_error_m": 1.198, "rotation_error_deg": 0.101, "psnr": 22.57, "fvd": 314.59},
        {
            "method": "MeshToken (ours)",
            "translation_error_m": cfg.traj100_translation_error_m,
            "rotation_error_deg": cfg.traj100_rotation_error_deg,
            "psnr": cfg.traj100_psnr,
            "fvd": cfg.traj100_fvd,
        },
    ]


def table2_realisdance_val(cfg: MeshTokenConfig | None = None) -> dict[str, float]:
    cfg = cfg or MeshTokenConfig()
    return {
        "I2V_BG": cfg.realis_i2v_bg,
        "Background_Consist": cfg.realis_bg_consist,
        "Aesthetic_Quality": cfg.realis_aesthetic,
    }


def table3_edited_motions() -> dict[str, dict[str, float]]:
    """Table 3 — edited Walking / Jogging (highlights)."""
    return {
        "Walking": {
            "MeshToken_Temporal_Flicker": 97.32,
            "MeshToken_Motion_Smooth": 98.79,
            "MeshToken_BG_Consist": 95.79,
        },
        "Jogging": {
            "MeshToken_Temporal_Flicker": 97.58,
            "MeshToken_Motion_Smooth": 98.34,
            "MeshToken_BG_Consist": 95.16,
        },
    }


def table4_ablation() -> list[dict[str, Any]]:
    return [
        {"ablation": "No Trajectory Input", "psnr": 15.32, "lpips": 0.2941},
        {"ablation": "No Pose Input", "psnr": 15.61, "lpips": 0.3131},
        {"ablation": "MLP for Mesh Tokenization", "psnr": 14.79, "lpips": 0.3271},
        {"ablation": "SMPL Parameters as Motion", "psnr": 15.91, "lpips": 0.2899},
        {"ablation": "No Per-Frame Cross Attention", "psnr": 16.49, "lpips": 0.2591},
        {"ablation": "MeshToken (ours)", "psnr": 16.78, "lpips": 0.2451},
    ]


def forward_smoke(cfg: MeshTokenConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MeshTokenConfig()
    motion = smpl_motion_stub(frames=cfg.latent_frames)
    tok = MotionTokenizer(cfg)
    zm = tok(
        canonical_vertices=motion["canonical"],
        gamma_world=motion["gamma_world"],
        rot_world=motion["rot_world"],
        gamma_cam=motion["gamma_cam"],
        rot_cam=motion["rot_cam"],
    )
    return {
        "motion_tokens_shape": list(zm.shape),
        "mesh_recon_mm": {"Trajectory100": cfg.mesh_recon_error_mm[0], "RealisDance-Val": cfg.mesh_recon_error_mm[1]},
        "dit": dit_motion_smoke(dim=128, tokens=32, motion_n=cfg.pose_tokens),
    }


def evaluation_demo(cfg: MeshTokenConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MeshTokenConfig()
    return {
        "framework": framework_card(cfg),
        "table1_trajectory100": table1_trajectory100(cfg),
        "table2_realisdance": table2_realisdance_val(cfg),
        "table3_edited": table3_edited_motions(),
        "table4_ablation": table4_ablation(),
        "forward": forward_smoke(cfg),
    }
