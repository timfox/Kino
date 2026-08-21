"""Paper tables (Winter et al., arXiv:2502.12691)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.sdt2i.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL

# Table 3 — main SDT2I comparison
TABLE3_MAIN = {
    "MD_original": {"task": "DT2I", "iou": 0.57, "ir": -0.47, "fid": 82.61},
    "MD_pano_lora": {"task": "DT2I", "iou": 0.66, "ir": -0.42, "fid": 60.99},
    "MSTD": {"task": "SDT2I", "iou": 0.67, "ir": -0.37, "fid": 61.12},
    "MPF_md_pano": {"task": "SDT2I", "iou": 0.45, "ir": -1.19, "fid": 84.60},
    "MPF_md_pers": {"task": "SDT2I", "iou": 0.05, "ir": -1.79, "fid": 107.00},
    "MPF_md_both": {"task": "SDT2I", "iou": 0.44, "ir": -1.22, "fid": 84.82},
}

# Table 4 — bootstrap coupling (MPF md both)
TABLE4_BOOTSTRAP_COUPLING = {
    "no": {"iou": 0.43, "ir": -1.26, "fid": 85.73},
    "yes": {"iou": 0.44, "ir": -1.22, "fid": 84.82},
}

# Table 6 — foreground EPPA off (MPF md both)
TABLE6_EPPA = {
    "fg_bg_eppa": {"iou": 0.44, "ir": -1.22, "fid": 84.82},
    "bg_eppa_only": {"iou": 0.52, "ir": -1.10, "fid": 82.21},
}

# Table 5 — global prompt in background
TABLE5_GLOBAL_PROMPT = {
    "MSTD_no_global": {"iou": 0.67, "fid": 61.12},
    "MSTD_global": {"iou": 0.59, "fid": 57.03},
}

DSYNVIEW_SCENES = 6
DSYNVIEW_SEEDS = 168
DSYNVIEW_PANORAMAS = 1008


def mstd_beats_mpf_iou() -> bool:
    return TABLE3_MAIN["MSTD"]["iou"] > TABLE3_MAIN["MPF_md_both"]["iou"]


def mstd_matches_md_baseline() -> bool:
    m, d = TABLE3_MAIN["MSTD"], TABLE3_MAIN["MD_pano_lora"]
    return abs(m["iou"] - d["iou"]) < 0.02 and abs(m["fid"] - d["fid"]) < 2.0


def bg_eppa_improves_mpf() -> bool:
    return TABLE6_EPPA["bg_eppa_only"]["iou"] > TABLE6_EPPA["fg_bg_eppa"]["iou"]


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "arxiv": PAPER_ARXIV,
        "title": PAPER_TITLE,
        "url": PAPER_URL,
        "table3": TABLE3_MAIN,
        "table4": TABLE4_BOOTSTRAP_COUPLING,
        "table5": TABLE5_GLOBAL_PROMPT,
        "table6": TABLE6_EPPA,
        "dsynview": {
            "scenes": DSYNVIEW_SCENES,
            "seeds": DSYNVIEW_SEEDS,
            "panoramas": DSYNVIEW_PANORAMAS,
        },
    }
