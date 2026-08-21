"""Paper tables (Shen et al., arXiv:2502.05902, AAAI 2025)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.faor.config import CODE_URL, PAPER_ARXIV, PAPER_TITLE, PAPER_URL, SAFE_BLOCKS

# Table 1 — ODI-SR test set
TABLE1_ODI_SR = {
    "Cubic": {2: (27.61, 0.8156), 4: (24.95, 0.6923), 8: (19.64, 0.5908), 16: (17.12, 0.4332)},
    "LIIF": {2: (27.34, 0.8214), 4: (22.29, 0.6626), 8: (19.45, 0.5692), 16: (17.59, 0.5231)},
    "OPE-SR": {2: (29.20, 0.8522), 4: (26.48, 0.7435), 8: (24.50, 0.6543), 16: (22.82, 0.5992)},
    "360-SS": {2: (27.14, 0.8095), 4: (23.20, 0.6613), 8: (21.65, 0.6417), 16: (19.65, 0.5431)},
    "LAU-Net": {2: (29.33, 0.8633), 4: (26.34, 0.7352), 8: (24.36, 0.6602), 16: (22.07, 0.5901)},
    "OSRT": {2: (29.61, 0.8700), 4: (26.26, 0.7443), 8: (24.24, 0.6532), 16: (22.49, 0.6030)},
    "FAOR": {2: (30.12, 0.8796), 4: (26.69, 0.7560), 8: (24.58, 0.6581), 16: (22.78, 0.6008)},
}

TABLE1_SUN360 = {
    "OSRT": {2: (31.20, 0.8958), 8: (24.24, 0.6533)},
    "FAOR": {2: (32.06, 0.9096), 8: (24.77, 0.6914)},
}

# Table 2 — SAFE priors ablation (ODI-SR)
TABLE2_PRIORS = {
    "FAOR": {8: (24.58, 0.6581)},
    "w/o_Md": {8: (24.47, 0.6575)},
    "w/o_Md_Ms": {8: (24.35, 0.6502)},
}

# Table 3 — geodesic resampling ablation
TABLE3_GEODESIC = {
    "FAOR": {8: (24.58, 0.6581)},
    "w/o_sphere": {8: (24.42, 0.6533)},
}

INFERENCE_SPEEDUP_VS_OSRT = 1.8  # illustrative; paper Fig. 1(b)


def faor_beats_osrt_odisr(scale: int = 8) -> bool:
    f = TABLE1_ODI_SR["FAOR"][scale][0]
    o = TABLE1_ODI_SR["OSRT"][scale][0]
    return f > o


def faor_beats_osrt_sun360(scale: int = 8) -> bool:
    return TABLE1_SUN360["FAOR"][scale][0] > TABLE1_SUN360["OSRT"][scale][0]


def geodesic_ablation_gain() -> float:
    return TABLE3_GEODESIC["FAOR"][8][0] - TABLE3_GEODESIC["w/o_sphere"][8][0]


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "arxiv": PAPER_ARXIV,
        "title": PAPER_TITLE,
        "url": PAPER_URL,
        "code": CODE_URL,
        "safe_blocks_paper": SAFE_BLOCKS,
        "table1_odisr": TABLE1_ODI_SR,
        "table1_sun360": TABLE1_SUN360,
        "table2_priors": TABLE2_PRIORS,
        "table3_geodesic": TABLE3_GEODESIC,
    }
