"""Pantheon360 framework card, paper tables, and smoke demos (arXiv:2605.25449)."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.pantheon360.cache import build_cache_from_erp, render_trajectory_erp
from ltx_trainer.pantheon360.conditioning import (
    clip_perspective_crops,
    concat_geometric_latent,
    diffusion_loss_stub,
    encode_geometry_scaffold,
    semantic_condition_vector,
)
from ltx_trainer.pantheon360.config import Pantheon360Config
from ltx_trainer.pantheon360.fusion import dual_anchor_latent_fusion
from ltx_trainer.pantheon360.integration import proceduralsky_card


def framework_card(cfg: Pantheon360Config | None = None) -> dict[str, Any]:
    cfg = cfg or Pantheon360Config()
    ps = proceduralsky_card()
    return {
        "name": "Pantheon360",
        "paper": cfg.paper_arxiv,
        "project": cfg.project_page,
        "backbone": "SVD U-Net (Argus init)",
        "recon": cfg.recon_backend,
        "conditioning": [
            "3D Cache → V_geo ERP render",
            "v_equi latent concat",
            "8× CLIP perspective crops",
            "Dual-anchor latent fusion (interpolation)",
        ],
        "applications": [
            "Street View interpolation",
            "360° video stabilization",
            "Digital-twin scene motion",
        ],
        "erp_resolution": f"{cfg.erp_height}×{cfg.erp_width}",
        "training_data": "360-1M filtered (~55k clips) + ViPE trajectory annotation",
        "integration": (
            "Trajectory-controlled 360° video complements proceduralsky HDR environment "
            f"maps ({ps['partner']}); joint GOPEX path via kino-procedural-skies-360.sh + fold weights."
        ),
        "proceduralsky": ps,
    }


def _metrics(
    fvd: float,
    ssim: float,
    psnr: float,
    lpips: float,
    met3r: float,
) -> dict[str, float]:
    return {"fvd": fvd, "ssim": ssim, "psnr": psnr, "lpips": lpips, "met3r": met3r}


def table_web360_single_view() -> dict[str, dict[str, float]]:
    """Table 1 — single 360° view-to-video on Web360."""
    return {
        "ViewCrafter": _metrics(525.746, 0.371, 15.654, 0.284, 0.4914),
        "TrajectoryCrafter": _metrics(517.475, 0.454, 15.151, 0.219, 0.4578),
        "GEN3C": _metrics(380.080, 0.583, 20.730, 0.145, 0.3496),
        "Pantheon360": _metrics(356.151, 0.746, 22.838, 0.065, 0.2840),
    }


def table_habitat_sparse_views() -> dict[str, dict[str, float]]:
    """Table 2 — sparse 360° views-to-video on Habitat."""
    return {
        "ViewCrafter": _metrics(778.207, 0.193, 11.833, 0.398, 0.5061),
        "TrajectoryCrafter": _metrics(690.322, 0.216, 12.223, 0.461, 0.6741),
        "GEN3C": _metrics(511.039, 0.481, 17.307, 0.195, 0.4522),
        "Pantheon360": _metrics(450.696, 0.756, 20.392, 0.091, 0.3026),
    }


def table_latent_fusion_ablation() -> dict[str, dict[str, float]]:
    """Table 3 — dual-anchor latent fusion (Google Street View)."""
    return {
        "Single": {"stwe": 0.124, "ie": 4.784, "psnr": 20.921, "ssim": 0.661, "lpips": 0.271},
        "Single+Latent Fusion": {"stwe": 0.420, "ie": 12.083, "psnr": 28.006, "ssim": 0.817, "lpips": 0.112},
        "Dual": {"stwe": 0.419, "ie": 8.120, "psnr": 27.860, "ssim": 0.817, "lpips": 0.093},
        "Dual+Latent Fusion": {"stwe": 0.395, "ie": 7.437, "psnr": 28.948, "ssim": 0.830, "lpips": 0.088},
    }


def table_cache_ablation_web360() -> dict[str, dict[str, float]]:
    """Supp. Table 4 — 3D Cache drop ratio on Web360."""
    return {
        "0%": _metrics(356.2, 0.746, 22.84, 0.065, 0.284),
        "25%": _metrics(382.6, 0.708, 22.10, 0.087, 0.318),
        "50%": _metrics(427.9, 0.643, 20.76, 0.124, 0.372),
        "75%": _metrics(496.4, 0.539, 18.85, 0.178, 0.446),
        "100% (w/o V_geo)": _metrics(553.3, 0.421, 16.93, 0.251, 0.523),
    }


def table_cache_ablation_habitat() -> dict[str, dict[str, float]]:
    """Supp. Table 5 — 3D Cache drop ratio on Habitat."""
    return {
        "0%": _metrics(450.7, 0.756, 20.39, 0.091, 0.303),
        "25%": _metrics(489.2, 0.712, 19.55, 0.118, 0.349),
        "50%": _metrics(551.8, 0.638, 18.12, 0.162, 0.414),
        "75%": _metrics(637.5, 0.524, 16.28, 0.231, 0.502),
        "100% (w/o V_geo)": _metrics(724.2, 0.387, 14.22, 0.319, 0.597),
    }


def table_runtime_a100() -> dict[str, dict[str, float | int]]:
    """Supp. Table 6 — runtime (s) and memory on single A100 @ 1024×512."""
    return {
        "Single-view": {
            "input_views": 1,
            "frames": 25,
            "recon_s": 34,
            "render_s": 2,
            "diffusion_s": 163,
            "total_s": 199,
            "mem_gb": 30,
        },
        "Interp. (latent fusion)": {
            "input_views": 2,
            "frames": 25,
            "recon_s": 50,
            "render_s": 5,
            "diffusion_s": 320,
            "total_s": 375,
            "mem_gb": 41,
        },
        "Long traj. (latent fusion)": {
            "input_views": 5,
            "frames": 100,
            "recon_s": 74,
            "render_s": 7,
            "diffusion_s": 1284,
            "total_s": 1365,
            "mem_gb": 41,
        },
    }


def benchmarks_bundle() -> dict[str, Any]:
    """All paper tables for CLI / AIML export."""
    return {
        "web360_single_view": table_web360_single_view(),
        "habitat_sparse_views": table_habitat_sparse_views(),
        "latent_fusion_ablation": table_latent_fusion_ablation(),
        "cache_ablation_web360": table_cache_ablation_web360(),
        "cache_ablation_habitat": table_cache_ablation_habitat(),
        "runtime_a100": table_runtime_a100(),
    }


def paper_checks() -> dict[str, bool]:
    """Sanity checks against published table ordering."""
    w360 = table_web360_single_view()
    hab = table_habitat_sparse_views()
    fusion = table_latent_fusion_ablation()
    cache_w = table_cache_ablation_web360()
    cache_h = table_cache_ablation_habitat()
    return {
        "beats_gen3c_web360_fvd": w360["Pantheon360"]["fvd"] < w360["GEN3C"]["fvd"],
        "beats_gen3c_web360_met3r": w360["Pantheon360"]["met3r"] < w360["GEN3C"]["met3r"],
        "beats_gen3c_habitat_fvd": hab["Pantheon360"]["fvd"] < hab["GEN3C"]["fvd"],
        "beats_gen3c_habitat_met3r": hab["Pantheon360"]["met3r"] < hab["GEN3C"]["met3r"],
        "dual_fusion_best_psnr": fusion["Dual+Latent Fusion"]["psnr"]
        >= max(v["psnr"] for v in fusion.values()),
        "dual_fusion_ie_vs_dual": fusion["Dual+Latent Fusion"]["ie"] < fusion["Dual"]["ie"],
        "vgeo_monotone_fvd_web360": cache_w["0%"]["fvd"] < cache_w["100% (w/o V_geo)"]["fvd"],
        "vgeo_monotone_met3r_habitat": cache_h["0%"]["met3r"] < cache_h["100% (w/o V_geo)"]["met3r"],
    }


def training_step_demo(cfg: Pantheon360Config | None = None) -> dict[str, float]:
    """Smoke: ERP cache → V_geo → latent concat + CLIP crops + fusion."""
    cfg = cfg or Pantheon360Config()
    torch.manual_seed(25449)
    frame = torch.rand(cfg.erp_height, cfg.erp_width, 3)
    cache = build_cache_from_erp(frame)
    v_geo = render_trajectory_erp(
        cache,
        torch.eye(4),
        t_frames=cfg.num_frames,
        height=cfg.erp_height,
        width=cfg.erp_width,
    )
    v_equi = encode_geometry_scaffold(v_geo)
    y_t = torch.randn(cfg.num_frames, v_equi.shape[1], 32, 64)
    _ = concat_geometric_latent(y_t, v_equi)

    crops = clip_perspective_crops(frame, num_crops=cfg.clip_yaw_crops)
    c_img = semantic_condition_vector(crops)

    eps = torch.randn_like(y_t)
    loss = float(diffusion_loss_stub(eps, eps * 0.9 + 0.1 * torch.randn_like(eps)).detach())

    x_fwd = torch.randn_like(y_t)
    x_bwd = torch.randn_like(y_t)
    fused = dual_anchor_latent_fusion(x_fwd, x_bwd)

    return {
        "v_geo_frames": float(v_geo.shape[0]),
        "latent_channels": float(v_equi.shape[1]),
        "clip_crop_count": float(crops.shape[0]),
        "c_img_dim": float(c_img.numel()),
        "diffusion_loss": loss,
        "fused_shape_match": float(fused.shape == y_t.shape),
    }


def evaluation_demo(cfg: Pantheon360Config | None = None) -> dict[str, Any]:
    cfg = cfg or Pantheon360Config()
    step = training_step_demo(cfg)
    checks = paper_checks()
    w360 = table_web360_single_view()

    return {
        **step,
        **checks,
        "pantheon360_web360_psnr": w360["Pantheon360"]["psnr"],
        "proceduralsky_partner": proceduralsky_card()["partner"],
    }
