"""Paper tables (Shen et al., arXiv:2504.09062)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.tpgs.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL

# Table I — OmniScenes (10 min / 100 min averages)
TABLE1_OMNISCENES = {
    "10min": {
        "3DGS(P)": {"PSNR": 23.80, "SSIM": 0.8424, "LPIPS": 0.2977},
        "ODGS": {"PSNR": 24.42, "SSIM": 0.8526, "LPIPS": 0.1391},
        "Ours": {"PSNR": 24.88, "SSIM": 0.8659, "LPIPS": 0.1362},
    },
    "100min": {
        "3DGS(P)": {"PSNR": 18.08, "SSIM": 0.7329, "LPIPS": 0.3600},
        "ODGS": {"PSNR": 24.51, "SSIM": 0.8505, "LPIPS": 0.1282},
        "Ours": {"PSNR": 25.13, "SSIM": 0.8614, "LPIPS": 0.1232},
    },
    "room_3_10min": {
        "ODGS": {"PSNR": 24.08, "SSIM": 0.8743, "LPIPS": 0.1384},
        "Ours": {"PSNR": 26.22, "SSIM": 0.9016, "LPIPS": 0.1256},
    },
}

# Table II — Ricoh360
TABLE2_RICOH360 = {
    "10min": {
        "3DGS(P)": {"PSNR": 25.12, "SSIM": 0.7932, "LPIPS": 0.2397},
        "ODGS": {"PSNR": 24.94, "SSIM": 0.8135, "LPIPS": 0.1489},
        "Ours": {"PSNR": 26.40, "SSIM": 0.8532, "LPIPS": 0.1581},
    },
    "100min": {
        "ODGS": {"PSNR": 25.44, "SSIM": 0.8260, "LPIPS": 0.1306},
        "Ours": {"PSNR": 26.94, "SSIM": 0.8554, "LPIPS": 0.1305},
    },
    "center_10min": {
        "3DGS(P)": {"PSNR": 27.24, "SSIM": 0.8364, "LPIPS": 0.2887},
        "ODGS": {"PSNR": 28.10, "SSIM": 0.8710, "LPIPS": 0.1206},
        "Ours": {"PSNR": 29.10, "SSIM": 0.8957, "LPIPS": 0.1041},
    },
}

# Table III — OmniPhotos
TABLE3_OMNIPHOTOS = {
    "10min": {
        "ODGS": {"PSNR": 26.24, "SSIM": 0.8704, "LPIPS": 0.1108},
        "Ours": {"PSNR": 26.98, "SSIM": 0.8777, "LPIPS": 0.1158},
    },
    "100min": {
        "Ours": {"PSNR": 27.74, "SSIM": 0.8903, "LPIPS": 0.0899},
    },
}

# Table IV — ablation (100 min PSNR / SSIM / LPIPS)
TABLE4_ABLATION = {
    "center": {
        "3DGS(P)": {"PSNR": 27.24, "SSIM": 0.8364, "LPIPS": 0.2887},
        "TP": {"PSNR": 28.41, "SSIM": 0.8803, "LPIPS": 0.1333},
        "TP+OP": {"PSNR": 29.08, "SSIM": 0.8965, "LPIPS": 0.0940},
        "TP+OP+CP": {"PSNR": 29.10, "SSIM": 0.8957, "LPIPS": 0.1041},
        "full_100min": {"PSNR": 29.72, "SSIM": 0.9034, "LPIPS": 0.0827},
    },
    "BeihaiPark": {
        "3DGS(P)": {"PSNR": 23.99, "SSIM": 0.8126, "LPIPS": 0.2600},
        "TP+OP+CP": {"PSNR": 26.05, "SSIM": 0.8848, "LPIPS": 0.1193},
        "full_100min": {"PSNR": 26.77, "SSIM": 0.8953, "LPIPS": 0.0904},
    },
    "room_3": {
        "TP+OP+CP": {"PSNR": 26.22, "SSIM": 0.9016, "LPIPS": 0.1256},
        "full_100min": {"PSNR": 26.56, "SSIM": 0.8993, "LPIPS": 0.1106},
    },
}

DATASETS = ("OmniScenes", "Ricoh360", "OmniPhotos")
BASELINES = ("NeRF(P)", "3DGS(P)", "TensoRF", "EgoNeRF", "ODGS", "Ours")


def ours_beats_odgs(dataset: str, split: str = "10min", metric: str = "PSNR") -> bool:
    tables = {
        "OmniScenes": TABLE1_OMNISCENES,
        "Ricoh360": TABLE2_RICOH360,
        "OmniPhotos": TABLE3_OMNIPHOTOS,
    }
    row = tables[dataset][split]
    return row["Ours"][metric] > row["ODGS"][metric]


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "url": PAPER_URL,
        "table1_omniscenes": TABLE1_OMNISCENES,
        "table2_ricoh360": TABLE2_RICOH360,
        "table3_omniphotos": TABLE3_OMNIPHOTOS,
        "table4_ablation": TABLE4_ABLATION,
        "datasets": DATASETS,
        "baselines": BASELINES,
    }
