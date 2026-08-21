"""Framework card and benchmarks for LiFT."""

from __future__ import annotations

from typing import Any

from ltx_trainer.lift.config import LiftConfig
from ltx_trainer.lift.layout import LIMITATIONS, LIFT_C_STAGES, LIFT_U_STAGES, PIPELINE_STAGES
from ltx_trainer.lift.mock import evaluation_smoke
from ltx_trainer.lift.tables import (
    fusion_heatmap_mlp,
    table1_mlp_per_view,
    table1_unconditional_fid,
    table2_missing_mr,
    table3_mr_to_ct,
    table4_through_plane_mrct,
    training_hyperparameters,
)


def framework_card(cfg: LiftConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LiftConfig()
    return {
        "name": cfg.paper_title,
        "arxiv": f"arXiv:{cfg.paper_arxiv}",
        "authors": "Zhang, Zhang, Jin, Marin-Llobet, Li, Quanzheng Li (Harvard / MGH)",
        "task": (
            "Factorize 3D medical volume synthesis into per-slice 2D generation plus "
            "lightweight inter-slice feature trajectories."
        ),
        "variants": {
            cfg.variant_unconditional: (
                "Frozen 2D generator + depth mapper; tri-planar distributional drifting (BraTS unconditional)."
            ),
            cfg.variant_conditional: (
                "2D translator + bidirectional GRU z-context; supervised Lpixel + MS-SSIM + ∆z (missing-MR, MR→CT)."
            ),
        },
        "datasets": list(cfg.datasets),
        "tasks": {
            "A_unconditional": {
                "dataset": "BraTS 2023 GLI",
                "resolution": list(cfg.lift_u_resolution),
                "headline": {
                    "fid_x1e3": cfg.lift_u_best_fid_x1e3,
                    "ms_ssim": cfg.lift_u_ms_ssim,
                    "inference_mem_gb": cfg.lift_u_inference_mem_gb,
                },
                "mapper_params_m": cfg.lift_u_mapper_params_m,
            },
            "B_missing_mr": {
                "dataset": "BraTS 2023 GLI (cWDM protocol)",
                "resolution": list(cfg.missing_mr_resolution),
                "validation_n": cfg.missing_mr_validation_n,
                "lift_c_inference_s": cfg.lift_c_inference_s,
                "cwdm_inference_s": cfg.cwdm_inference_s,
                "speedup_ratio": cfg.speedup_vs_cwdm,
            },
            "C_mr_to_ct": {
                "dataset": "SynthRAD2023 Task 1 Brain",
                "split": f"{cfg.synthrad_train} train / {cfg.synthrad_test} test",
                "resolution": list(cfg.mrct_resolution),
                "mae_hu": cfg.lift_c_mae_hu,
                "psnr": cfg.lift_c_psnr,
                "ssim": cfg.lift_c_ssim,
                "ncc": cfg.lift_c_ncc,
                "dz_mae_full": cfg.lift_c_dz_mae_full,
                "dz_corr": cfg.lift_c_dz_corr,
            },
        },
        "findings": [
            "Dependent posterior views (L3, R3) strongest per-view signal.",
            "Temporal difference beats concatenation for inter-slice features.",
            "Feature concatenation + temporal difference best fusion (F1 0.80 MLP, All Views).",
            "LiFT-C approaches cWDM missing-MR quality at ~135× lower inference (setup-specific).",
            "MR-to-CT gains larger on ∆z metrics than voxel MAE when BiGRU mapper is added.",
        ],
        "pipeline_stages": list(PIPELINE_STAGES),
        "lift_u_stages": list(LIFT_U_STAGES),
        "lift_c_stages": list(LIFT_C_STAGES),
        "tri_planes": list(cfg.tri_planes),
        "limitations": list(LIMITATIONS),
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_unconditional_fid": table1_unconditional_fid(),
        "table1_mlp_per_view_excerpt": table1_mlp_per_view(),
        "fig3_fusion_heatmap": fusion_heatmap_mlp(),
        "table2_missing_mr": table2_missing_mr(),
        "table3_mr_to_ct": table3_mr_to_ct(),
        "table4_through_plane": table4_through_plane_mrct(),
        "table10_training": training_hyperparameters(),
    }


def evaluation_demo() -> dict[str, Any]:
    return {"smoke": evaluation_smoke(), "headline_fid_x1e3": LiftConfig().lift_u_best_fid_x1e3}


def headline_results() -> dict[str, Any]:
    cfg = LiftConfig()
    return {
        "lift_u_fid_x1e3": cfg.lift_u_best_fid_x1e3,
        "lift_u_mem_gb": cfg.lift_u_inference_mem_gb,
        "lift_c_missing_mr_s": cfg.lift_c_inference_s,
        "lift_c_mrct_mae_hu": cfg.lift_c_mae_hu,
        "lift_c_dz_corr": cfg.lift_c_dz_corr,
    }
