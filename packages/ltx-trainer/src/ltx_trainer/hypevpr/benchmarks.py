"""Paper tables (Woo et al., arXiv:2506.04764)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.hypevpr.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL

# Table 1 — Pitts250K-P2E / YQ360 (R@1 %, time/q ms) — selected rows
TABLE1_P2E = {
    "PanoVPRx16_SwinT": {"time_ms": 48.6, "pitts_r1": 33.6, "yq360_r1": 43.2},
    "HypeVPR-L_SwinT": {"time_ms": 14.0, "pitts_r1": 32.5, "yq360_r1": 45.6},
    "HypeVPR-B_ConvNeXtS": {"time_ms": 3.6, "pitts_r1": 34.3, "yq360_r1": 43.8},
    "HypeVPR-B_star_ResNet50": {"time_ms": 29.6, "pitts_r1": 79.6, "yq360_r1": 63.6},
    "HypeVPR-O_star": {"time_ms": 4.0, "pitts_r1": 66.5, "yq360_r1": 53.6},
    "Orhan_star": {"time_ms": 1555.2, "pitts_r1": 47.0, "yq360_r1": 47.6},
}

# Table 2 — Pitts250k-P2E vs P2P SOTA (fine-tuned *)
TABLE2_PITTS = {
    "EigenPlace": {"time_ms": 90.5, "storage_mb": 262.4, "desc": 2048, "r1": 78.3},
    "EigenPlace_star": {"time_ms": 90.5, "storage_mb": 262.4, "desc": 2048, "r1": 80.9},
    "SALAD": {"time_ms": 371.4, "storage_mb": 1082.4, "desc": 8448, "r1": 86.8},
    "HypeVPR-O_star": {"time_ms": 6.1, "storage_mb": 16.4, "desc": 2048, "r1": 66.5},
    "HypeVPR-SW_star": {"time_ms": 90.5, "storage_mb": 262.4, "desc": 2048, "r1": 80.6},
    "HypeVPR-B_star": {"time_ms": 17.9, "storage_mb": 147.6, "desc": 2048, "r1": 79.6},
    "HypeVPR-L_star": {"time_ms": 41.8, "storage_mb": 278.8, "desc": 2048, "r1": 81.2},
}

# Table 3 — SF-XL panoramic test DB
TABLE3_SFXL = {
    "EigenPlace": {"time_s": 7.92, "storage_gb": 21.4, "desc": 2048, "r1": 84.1},
    "SALAD": {"time_s": 32.69, "storage_gb": 88.3, "desc": 8448, "r1": 88.6},
    "HypeVPR-B": {"time_s": 1.79, "storage_gb": 16.0, "desc": 2048, "r1": 80.5},
    "HypeVPR-L": {"time_s": 2.92, "storage_gb": 30.3, "desc": 2048, "r1": 85.2},
}

# Table 4 — 2-level Euclidean vs Poincaré
TABLE4_MANIFOLD = {
    "Euclidean": {"r1": 9.2, "r5": 20.9, "r10": 28.6, "r20": 36.9},
    "Poincare": {"r1": 14.9, "r5": 30.8, "r10": 41.9, "r20": 50.8},
}

# Table 5 — loss ablation (HypeVPR-O)
TABLE5_LOSS = {
    "w/o_Leuc": {"r1": 32.0},
    "w/o_Lhyp": {"r1": 64.3},
    "w/o_Lhier": {"r1": 50.8},
    "full": {"r1": 66.5},
}

VARIANT_LEVELS = {
    "HypeVPR-O": (1,),
    "HypeVPR-B": (1, 4),
    "HypeVPR-L": (1, 5),
    "HypeVPR-SW": (5,),
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "url": PAPER_URL,
        "table1_p2e": TABLE1_P2E,
        "table2_pitts": TABLE2_PITTS,
        "table3_sfxl": TABLE3_SFXL,
        "table4_manifold": TABLE4_MANIFOLD,
        "table5_loss": TABLE5_LOSS,
        "variant_levels": VARIANT_LEVELS,
    }
