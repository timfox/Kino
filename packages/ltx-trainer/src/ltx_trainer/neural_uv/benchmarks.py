"""Paper tables and anchors (Salehi arXiv:2606.10050)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.neural_uv.config import (
    COMPACT_CHARTS,
    PAPER_ARXIV,
    PAPER_AUTHORS,
    PAPER_REPO,
    PAPER_TITLE,
    PAPER_URL,
    PAPER_VENUE,
)

# Table 1 — compact pre-cut charts (Sec. 5)
TABLE1_COMPACT: list[dict[str, Any]] = [
    {"mesh": "Hand", "method": "LSCM", "esd": 1.47e9, "conf": 8.98, "area": 27.71, "flip_pct": 13.56, "time_s": 0.001},
    {"mesh": "Hand", "method": "ARAP", "esd": 1262.61, "conf": 10.47, "area": 1.58, "flip_pct": 15.74, "time_s": 0.004},
    {"mesh": "Hand", "method": "SLIM-1k", "esd": 7.86e7, "conf": 7.85, "area": 27.54, "flip_pct": 13.56, "time_s": 1.194},
    {"mesh": "Hand", "method": "BFF", "esd": 6.20e7, "conf": 8.35, "area": 27.59, "flip_pct": 13.80, "time_s": 0.305},
    {"mesh": "Hand", "method": "Ours", "esd": 12.73, "conf": 4.68, "area": 0.73, "flip_pct": 0.00, "time_s": 23.44},
    {"mesh": "Hand", "method": "OptCuts", "esd": 4.10, "conf": 1.16, "area": 0.02, "flip_pct": 0.00, "time_s": 1.755, "re_cut": True},
    {"mesh": "Bob", "method": "BFF", "esd": 14.04, "conf": 1.24, "area": 1.08, "flip_pct": 0.00, "time_s": 0.305},
    {"mesh": "Bob", "method": "Ours", "esd": 4.39, "conf": 1.30, "area": 0.07, "flip_pct": 0.00, "time_s": 37.45},
    {"mesh": "Camel", "method": "SLIM-1k", "esd": 4.09, "conf": 1.14, "area": 0.02, "flip_pct": 0.00, "time_s": 6.381},
    {"mesh": "Camel", "method": "BFF", "esd": 4.27, "conf": 1.10, "area": 0.17, "flip_pct": 0.00, "time_s": 0.348},
    {"mesh": "Camel", "method": "Ours", "esd": 4.14, "conf": 1.18, "area": 0.07, "flip_pct": 0.00, "time_s": 27.80},
    {"mesh": "Camel", "method": "OptCuts", "esd": 4.09, "conf": 1.15, "area": 0.02, "flip_pct": 0.00, "time_s": 0.110, "re_cut": True},
]

# Table 2 — failure-mode check
TABLE2_FAILURE: list[dict[str, Any]] = [
    {"mesh": "Hand", "components": "1/1", "slim_1k": "13.56% flips", "bff": "13.80% flips", "optcuts": "ESD=4.10"},
    {"mesh": "Bob", "components": "3/3", "slim_1k": "init fail", "bff": "ESD=14.04", "optcuts": "cleanup fail"},
    {"mesh": "Camel", "components": "1/1", "slim_1k": "ESD=4.09", "bff": "ESD=4.27", "optcuts": "ESD=4.09"},
]

# Table 3 — compact ablations
TABLE3_ABLATION: list[dict[str, Any]] = [
    {"variant": "Full (ω0=15, k=16)", "esd": 7.09, "flip_pct": 0.00, "min_det": 0.27},
    {"variant": "No spectral (k=0)", "esd": 8.01, "flip_pct": 0.00, "min_det": 0.19},
    {"variant": "Spectral init invalid", "esd": None, "flip_pct": 49.70, "min_det": -11.78},
    {"variant": "Random init invalid", "esd": None, "flip_pct": 50.60, "min_det": -33.47},
]

# Table 5 — Thingi10K scale
TABLE5_THINGI: list[dict[str, Any]] = [
    {"subset": "Raw usable", "tried": 26, "finite": 18, "valid": 18, "valid_pct": 69.2, "face_range": "165–131072", "med_esd": 5.27},
    {"subset": "Xatlas first-500", "tried": 287, "finite": 283, "valid": 280, "valid_pct": 97.6, "face_range": "4–236", "med_esd": 5.45},
    {"subset": "Xatlas stratified", "tried": 47, "finite": 47, "valid": 42, "valid_pct": 89.4, "face_range": "1028–196608", "med_esd": 60.56},
]

# Table 6 — runtime Pareto (compact avg)
TABLE6_PARETO: list[dict[str, Any]] = [
    {"set": "Compact", "iters": 500, "valid": "3/3", "med_esd": 5.78, "med_time_s": 3.21},
    {"set": "Compact", "iters": 1000, "valid": "3/3", "med_esd": 4.42, "med_time_s": 5.88},
    {"set": "Compact", "iters": 5000, "valid": "3/3", "med_esd": 4.39, "med_time_s": 24.18},
]

# Table 7 — Amara coverage
TABLE7_AMARA: list[dict[str, Any]] = [
    {"method": "Ours neural-only", "coverage_pct": 52.1, "valid_charts": "4708/4708", "flips": 0},
    {"method": "Ours full fallback", "coverage_pct": 100.0, "valid_charts": "0 local flips", "flips": 0},
    {"method": "Blender Smart UV", "coverage_pct": 100.0, "valid_charts": "246/1219095", "flips": 246},
]

PAPER_ANCHORS = {
    "compact_zero_flip_charts": 3,
    "thingi_stratified_valid": "42/47",
    "xatlas_first500_valid": "280/287",
    "amara_neural_charts_zero_flip": "4708/4708",
    "amara_rust_1k_zero_flip_faces": "48.5M",
}


def table1_ours_rows() -> list[dict[str, Any]]:
    return [r for r in TABLE1_COMPACT if r["method"] == "Ours"]


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": {
            "arxiv": PAPER_ARXIV,
            "title": PAPER_TITLE,
            "authors": PAPER_AUTHORS,
            "venue": PAPER_VENUE,
            "url": PAPER_URL,
            "repo": PAPER_REPO,
        },
        "compact_charts": list(COMPACT_CHARTS),
        "table1_compact": TABLE1_COMPACT,
        "table2_failure": TABLE2_FAILURE,
        "table3_ablation": TABLE3_ABLATION,
        "table5_thingi": TABLE5_THINGI,
        "table6_pareto": TABLE6_PARETO,
        "table7_amara": TABLE7_AMARA,
        "anchors": PAPER_ANCHORS,
    }
