"""Paper tables (Ito et al., arXiv:2505.19883)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.erpgs.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL

# Table 1 — NVS on OmniBlender, Ricoh360, OmniScenes (selected scenes)
TABLE1_NVS: dict[str, dict[str, dict[str, float]]] = {
    "OmniBlender": {
        "barbershop": {
            "EgoNeRF": {"PSNR": 30.57, "SSIM": 0.900, "LPIPS_A": 0.187},
            "ODGS": {"PSNR": 33.66, "SSIM": 0.947, "LPIPS_A": 0.123},
            "OmniGS": {"PSNR": 37.26, "SSIM": 0.974, "LPIPS_A": 0.050},
            "Ours": {"PSNR": 38.71, "SSIM": 0.979, "LPIPS_A": 0.040},
        },
        "lone-monk": {
            "EgoNeRF": {"PSNR": 31.10, "SSIM": 0.935, "LPIPS_A": 0.073},
            "ODGS": {"PSNR": 28.58, "SSIM": 0.922, "LPIPS_A": 0.098},
            "OmniGS": {"PSNR": 29.00, "SSIM": 0.943, "LPIPS_A": 0.067},
            "Ours": {"PSNR": 32.34, "SSIM": 0.963, "LPIPS_A": 0.037},
        },
        "archiviz-flat": {
            "EgoNeRF": {"PSNR": 31.69, "SSIM": 0.917, "LPIPS_A": 0.103},
            "ODGS": {"PSNR": 32.50, "SSIM": 0.943, "LPIPS_A": 0.095},
            "OmniGS": {"PSNR": 33.38, "SSIM": 0.954, "LPIPS_A": 0.056},
            "Ours": {"PSNR": 35.95, "SSIM": 0.963, "LPIPS_A": 0.040},
        },
        "classroom": {
            "EgoNeRF": {"PSNR": 26.75, "SSIM": 0.770, "LPIPS_A": 0.368},
            "ODGS": {"PSNR": 26.20, "SSIM": 0.798, "LPIPS_A": 0.385},
            "OmniGS": {"PSNR": 33.03, "SSIM": 0.906, "LPIPS_A": 0.190},
            "Ours": {"PSNR": 33.62, "SSIM": 0.917, "LPIPS_A": 0.157},
        },
    },
    "Ricoh360": {
        "bricks": {
            "EgoNeRF": {"PSNR": 24.39, "SSIM": 0.791, "LPIPS_A": 0.186},
            "ODGS": {"PSNR": 22.23, "SSIM": 0.724, "LPIPS_A": 0.293},
            "OmniGS": {"PSNR": 22.27, "SSIM": 0.766, "LPIPS_A": 0.251},
            "Ours": {"PSNR": 25.03, "SSIM": 0.820, "LPIPS_A": 0.173},
        },
        "center": {
            "EgoNeRF": {"PSNR": 29.42, "SSIM": 0.874, "LPIPS_A": 0.144},
            "ODGS": {"PSNR": 24.37, "SSIM": 0.789, "LPIPS_A": 0.396},
            "OmniGS": {"PSNR": 26.78, "SSIM": 0.855, "LPIPS_A": 0.188},
            "Ours": {"PSNR": 28.63, "SSIM": 0.879, "LPIPS_A": 0.138},
        },
    },
    "OmniScenes": {
        "room": {
            "EgoNeRF": {"PSNR": 28.69, "SSIM": 0.904, "LPIPS_A": 0.202},
            "ODGS": {"PSNR": 27.25, "SSIM": 0.900, "LPIPS_A": 0.225},
            "OmniGS": {"PSNR": 30.29, "SSIM": 0.928, "LPIPS_A": 0.155},
            "Ours": {"PSNR": 31.00, "SSIM": 0.932, "LPIPS_A": 0.142},
        },
    },
}

# Table 2 — ablation on OmniBlender (averages)
TABLE2_ABLATION = {
    "w/o W": {"PSNR": 33.86, "SSIM": 0.949, "LPIPS_A": 0.0852, "LPIPS_V": 0.1630},
    "w/o L_dn": {"PSNR": 34.61, "SSIM": 0.953, "LPIPS_A": 0.0717, "LPIPS_V": 0.1429},
    "w/o L_s": {"PSNR": 34.64, "SSIM": 0.954, "LPIPS_A": 0.0723, "LPIPS_V": 0.1434},
    "All": {"PSNR": 35.16, "SSIM": 0.956, "LPIPS_A": 0.0687, "LPIPS_V": 0.1374},
}

DATASETS = ("OmniBlender", "Ricoh360", "OmniScenes")
BASELINES = ("EgoNeRF", "ODGS", "OmniGS", "Ours")


def ours_beats_baseline(scene_metrics: dict[str, dict[str, float]], baseline: str) -> bool:
    ours = scene_metrics["Ours"]
    base = scene_metrics[baseline]
    return ours["PSNR"] > base["PSNR"] and ours["SSIM"] > base["SSIM"]


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "url": PAPER_URL,
        "table1_nvs": TABLE1_NVS,
        "table2_ablation": TABLE2_ABLATION,
        "datasets": list(DATASETS),
        "baselines": list(BASELINES),
    }
