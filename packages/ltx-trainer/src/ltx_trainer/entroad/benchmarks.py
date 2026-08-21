"""Paper table excerpts — cross-dataset ZSAD (Sec. 5)."""

from __future__ import annotations

from typing import Any

# Table 1 — image-level (AUROC, AP) average row + highlights
TABLE1_IMAGE: dict[str, tuple[float, float]] = {
    "WinCLIP": (80.0, 83.4),
    "AnomalyCLIP": (84.8, 87.3),
    "FiLo": (86.4, 89.4),
    "AA-CLIP": (85.5, 88.9),
    "FAPrompt": (90.5, 91.8),
    "MRAD": (91.1, 92.4),
    "EntroAD": (91.9, 93.8),
}

TABLE1_ENTROAD_ROWS: dict[str, tuple[float, float]] = {
    "MVTec-AD": (93.3, 97.2),
    "VisA": (84.7, 87.2),
    "MPDD": (79.7, 82.9),
    "BrainMRI": (96.1, 96.9),
    "HeadCT": (95.7, 96.4),
    "Br35H": (97.0, 97.0),
}

# Table 2 — pixel-level (AUROC, AUPRO) average
TABLE2_PIXEL: dict[str, tuple[float, float]] = {
    "WinCLIP": (73.8, 40.7),
    "AnomalyCLIP": (90.2, 75.9),
    "FiLo": (91.5, 65.0),
    "AA-CLIP": (92.0, 75.5),
    "FAPrompt": (91.8, 76.2),
    "MRAD": (93.4, 78.9),
    "EntroAD": (93.5, 80.1),
}

TABLE2_ENTROAD_ROWS: dict[str, tuple[float, float]] = {
    "MVTec-AD": (92.8, 86.2),
    "VisA": (94.8, 87.4),
    "DTD": (98.5, 92.6),
    "Endo": (89.4, 73.9),
    "Kvasir": (85.3, 53.5),
}

# Table 3 — ablation (selected columns)
TABLE3_ABLATION: dict[str, dict[str, tuple[float, float]]] = {
    "EntroAD": {
        "MVTec-AD_img": (93.3, 97.2),
        "MVTec-AD_pix": (92.8, 86.2),
        "Endo_pix": (89.4, 73.9),
    },
    "w/o Gating": {
        "MVTec-AD_img": (89.8, 95.3),
        "MVTec-AD_pix": (79.7, 42.8),
        "Endo_pix": (88.6, 72.9),
    },
    "w/o Entropy Routing": {
        "MVTec-AD_img": (93.0, 96.9),
        "MVTec-AD_pix": (92.7, 84.1),
    },
    "w/o Dual-branch": {
        "MVTec-AD_img": (93.1, 96.9),
        "MVTec-AD_pix": (92.4, 84.5),
        "Kvasir_pix": (82.9, 48.6),
    },
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_image_average": TABLE1_IMAGE,
        "table1_entroad": TABLE1_ENTROAD_ROWS,
        "table2_pixel_average": TABLE2_PIXEL,
        "table2_entroad": TABLE2_ENTROAD_ROWS,
        "table3_ablation": TABLE3_ABLATION,
        "entroad_beats_mrad_image_auroc": TABLE1_IMAGE["EntroAD"][0] > TABLE1_IMAGE["MRAD"][0],
        "entroad_beats_mrad_pixel_aupro": TABLE2_PIXEL["EntroAD"][1] > TABLE2_PIXEL["MRAD"][1],
        "gating_critical_mvtec_pix": TABLE3_ABLATION["w/o Gating"]["MVTec-AD_pix"][1] < 50.0,
    }
