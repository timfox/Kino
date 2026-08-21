"""Paper tables (Ito et al., arXiv:2505.20126)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.ob3d.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL

# Table 1 — dataset comparison (subset + OB3D)
TABLE1_DATASETS = {
    "Matterport3D": {"depth": True, "normal": True, "mesh_points": True, "protocol_3d": False},
    "OmniBlender": {"depth": False, "normal": False, "mesh_points": False, "protocol_3d": False},
    "OB3D": {"depth": True, "normal": True, "mesh_points": True, "protocol_3d": True},
}

# Table 3 — camera parameter estimation (averages)
TABLE3_CPE = {
    "indoor_OpenSfM_ego": {"RRA": 0.405, "RTA": 0.037, "AUC5": 1.000, "ATE": 0.000},
    "indoor_OpenMVG_non_ego": {"RRA": 0.413, "RTA": 0.058, "AUC5": 1.000, "ATE": 0.000},
    "outdoor_OpenMVG_non_ego": {"RRA": 12.386, "RTA": 6.443, "AUC5": 0.851, "ATE": 0.091},
}

# Table 4 — novel view synthesis (averages)
TABLE4_NVS = {
    "EgoNeRF_indoor_ego": {"PSNR": 32.26, "SSIM": 0.906, "LPIPS_A": 0.128},
    "OmniGS_indoor_ego": {"PSNR": 33.25, "SSIM": 0.897, "LPIPS_A": 0.169},
    "ODGS_outdoor_non_ego": {"PSNR": 25.04, "SSIM": 0.771, "LPIPS_A": 0.243},
    "op43dgs_indoor_non_ego": {"PSNR": 31.18, "SSIM": 0.905, "LPIPS_A": 0.154},
}

# Table 5 — 3D reconstruction (selected averages)
TABLE5_RECON = {
    "COLMAP_indoor_ego": {"RMSE": 0.978, "delta125": 0.905},
    "COLMAP_outdoor_non_ego": {"RMSE": 2.228, "delta125": 0.916},
    "NeuS_indoor_ego": {"RMSE": 0.406, "delta125": 0.989},
    "NeuS_all_non_ego": {"RMSE": 1.297, "delta125": 0.959},
    "NeuS_barbershop_non_ego": {"RMSE": 0.121, "delta125": 0.986},
    "NeuS_emerald_square_ego": {"RMSE": 6.235, "delta125": 0.951},
}

TABLE5_RECON_SCENES = {
    "archiviz-flat": {
        "COLMAP": {"ego": {"RMSE": 0.883, "d125": 0.688}, "non_ego": {"RMSE": 1.073, "d125": 0.635}},
        "NeuS": {"ego": {"RMSE": 0.206, "d125": 0.994}, "non_ego": {"RMSE": 0.191, "d125": 0.986}},
    },
    "barbershop": {
        "COLMAP": {"ego": {"RMSE": 0.277, "d125": 0.960}, "non_ego": {"RMSE": 0.240, "d125": 0.950}},
        "NeuS": {"ego": {"RMSE": 0.319, "d125": 0.991}, "non_ego": {"RMSE": 0.121, "d125": 0.986}},
    },
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "url": PAPER_URL,
        "table1_datasets": TABLE1_DATASETS,
        "table3_cpe": TABLE3_CPE,
        "table4_nvs": TABLE4_NVS,
        "table5_recon": TABLE5_RECON,
        "table5_scenes": TABLE5_RECON_SCENES,
    }
