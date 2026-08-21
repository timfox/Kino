"""Paper tables (Jung et al., arXiv:2502.20685)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.edm.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL

# Table 1 — Matterport3D pose AUC (%)
TABLE1_MATTERPORT3D = {
    "SPHORB": {"auc_5": 0.38, "auc_10": 1.41, "auc_20": 3.99, "feature": "sparse"},
    "SphereGlue": {"auc_5": 11.29, "auc_10": 19.95, "auc_20": 31.10, "feature": "sparse"},
    "DKM": {"auc_5": 18.43, "auc_10": 28.50, "auc_20": 38.44, "feature": "dense"},
    "RoMa": {"auc_5": 12.45, "auc_10": 22.37, "auc_20": 34.24, "feature": "dense"},
    "EDM": {"auc_5": 45.15, "auc_10": 60.99, "auc_20": 73.60, "feature": "dense"},
}

# Table 2 — Stanford2D3D
TABLE2_STANFORD2D3D = {
    "SPHORB": {"auc_5": 0.14, "auc_10": 1.01, "auc_20": 4.08},
    "SphereGlue": {"auc_5": 11.25, "auc_10": 22.41, "auc_20": 36.57},
    "DKM": {"auc_5": 12.46, "auc_10": 22.18, "auc_20": 34.13},
    "RoMa": {"auc_5": 11.48, "auc_10": 22.52, "auc_20": 37.07},
    "EDM": {"auc_5": 55.08, "auc_10": 71.65, "auc_20": 82.72},
}

# Table 3 — ablation (Matterport3D AUC %)
TABLE3_ABLATION = {
    "DKM*": {"auc_5": 19.83, "auc_10": 33.06, "auc_20": 46.24},
    "2D_linear_bidir": {"auc_5": 29.67, "auc_10": 45.90, "auc_20": 60.82},
    "2D_linear_bidir_rot": {"auc_5": 35.03, "auc_10": 51.14, "auc_20": 65.07},
    "3D_linear_bidir": {"auc_5": 34.64, "auc_10": 50.82, "auc_20": 65.16},
    "3D_linear_full": {"auc_5": 45.15, "auc_10": 60.99, "auc_20": 73.60},
    "3D_sinusoidal_full": {"auc_5": 42.39, "auc_10": 58.27, "auc_20": 70.98},
}

IMPROVEMENT_MP_AUC5 = 45.15 - 18.43  # +26.72 vs DKM
IMPROVEMENT_S2D_AUC5 = 55.08 - 12.46  # +42.62 vs DKM

DATABASES = ("Matterport3D", "Stanford2D3D", "EgoNeRF", "OmniPhotos")


def edm_beats_dkm_matterport() -> bool:
    return TABLE1_MATTERPORT3D["EDM"]["auc_5"] > TABLE1_MATTERPORT3D["DKM"]["auc_5"]


def edm_beats_all_s2d() -> bool:
    e = TABLE2_STANFORD2D3D["EDM"]["auc_5"]
    return e > max(TABLE2_STANFORD2D3D[k]["auc_5"] for k in TABLE2_STANFORD2D3D if k != "EDM")


def ablation_3d_full_best() -> bool:
    return TABLE3_ABLATION["3D_linear_full"]["auc_5"] >= TABLE3_ABLATION["3D_sinusoidal_full"]["auc_5"]


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "arxiv": PAPER_ARXIV,
        "title": PAPER_TITLE,
        "url": PAPER_URL,
        "table1": TABLE1_MATTERPORT3D,
        "table2": TABLE2_STANFORD2D3D,
        "table3": TABLE3_ABLATION,
        "improvement_matterport_auc5": IMPROVEMENT_MP_AUC5,
        "improvement_stanford_auc5": IMPROVEMENT_S2D_AUC5,
        "databases": list(DATABASES),
    }
