"""Reference metrics (Wu et al., arXiv:2601.16192)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.anything360.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL, PROJECT_URL

TABLE1_LAVAL_OURS: dict[str, Any] = {
    "dataset": "Laval Indoor",
    "FID": 8.0,
    "KID_x100": 0.22,
    "CLIP_FID": 4.6,
    "FAED": 9.8,
    "CLIP_score": 29.21,
}

TABLE1_SUN360_OURS: dict[str, Any] = {
    "dataset": "SUN360",
    "FID": 22.4,
    "KID_x100": 1.27,
    "CLIP_FID": 7.3,
    "FAED": 3.8,
    "CLIP_score": 28.07,
}

TABLE1_CUBEDIFF_LAVAL: dict[str, Any] = {"method": "CubeDiff", "FID": 9.5, "FAED": 18.4, "CLIP_FID": 3.2}

TABLE2_REAL_CAMERA: dict[str, Any] = {
    "split": "real_camera_trajectory",
    "PSNR": 25.75,
    "LPIPS": 0.0468,
    "FVD": 483.4,
    "VBench_imaging": 0.5515,
    "VBench_aesthetic": 0.5427,
    "VBench_motion": 0.9885,
}

TABLE2_SIM_CAMERA: dict[str, Any] = {
    "split": "simulated_camera_trajectory",
    "PSNR": 23.64,
    "LPIPS": 0.0846,
    "FVD": 432.9,
    "VBench_imaging": 0.5489,
    "VBench_aesthetic": 0.5394,
    "VBench_motion": 0.9891,
}

TABLE3_FOV_NYUv2: dict[str, Any] = {"dataset": "NYUv2", "mean_deg": 3.90, "median_deg": 3.17}

TABLE5_CLE_DS: dict[str, Any] = {
    "image": {"vanilla": 9.92, "blended_decoding": 5.29, "CLE": 3.87},
    "video": {"vanilla": 35.52, "blended_decoding": 19.84, "CLE": 13.28},
}

TABLE7_CANONICAL: dict[str, Any] = {
    "note": "canonical training vs no (256×512 ablation)",
    "FVD_real_no": 559.5,
    "FVD_real_yes": 470.8,
}


def table1_laval() -> dict[str, Any]:
    return dict(TABLE1_LAVAL_OURS)


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": PAPER_TITLE,
        "arxiv": f"arXiv:{PAPER_ARXIV}",
        "paper_url": PAPER_URL,
        "project_url": PROJECT_URL,
        "table1_laval_ours": TABLE1_LAVAL_OURS,
        "table1_sun360_ours": TABLE1_SUN360_OURS,
        "table1_cubediff_laval": TABLE1_CUBEDIFF_LAVAL,
        "table2_real_camera": TABLE2_REAL_CAMERA,
        "table2_sim_camera": TABLE2_SIM_CAMERA,
        "table3_fov_nyuv2": TABLE3_FOV_NYUv2,
        "table5_cle_ds": TABLE5_CLE_DS,
        "table7_canonical": TABLE7_CANONICAL,
    }
