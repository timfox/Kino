"""Paper Tables 4–7 (Benny & Wolf, arXiv:2412.06968)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.sphereuformer.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL

# Table 4 — rank 7 (256×512 equivalent), OURS = Tab. 3 config (1)
TABLE4_RANK7 = {
    "PanoFormer": {
        "params_m": 14.5,
        "flops_g": 11.8,
        "s2d3d": {"mae": 0.174, "mre": 0.078, "d1": 92.5},
        "s3d": {"mae": 0.154, "mre": 0.051, "d1": 94.8},
        "s2d3d_seg": {"acc": 83.1, "miou": 60.6},
        "s3d_seg": {"acc": 94.9, "miou": 49.7},
    },
    "EGFormer": {
        "params_m": 15.2,
        "flops_g": 15.6,
        "s2d3d": {"mae": 0.170, "mre": 0.075, "d1": 93.1},
        "s3d": {"mae": 0.150, "mre": 0.049, "d1": 95.2},
        "s2d3d_seg": {"acc": 86.5, "miou": 66.4},
        "s3d_seg": {"acc": 95.0, "miou": 51.5},
    },
    "Elite360D": {
        "params_m": 14.7,
        "flops_g": 13.6,
        "s2d3d": {"mae": 0.169, "mre": 0.069, "d1": 93.5},
        "s3d": {"mae": 0.147, "mre": 0.046, "d1": 95.9},
        "s2d3d_seg": {"acc": 87.4, "miou": 71.4},
        "s3d_seg": {"acc": 95.3, "miou": 52.0},
    },
    "MVDiffusion": {},
    "OURS": {
        "params_m": 14.9,
        "flops_g": 13.1,
        "nodes_k": 164,
        "s2d3d": {"mae": 0.165, "mre": 0.071, "d1": 94.0},
        "s3d": {"mae": 0.142, "mre": 0.045, "d1": 96.4},
        "s2d3d_seg": {"acc": 88.6, "miou": 72.2},
        "s3d_seg": {"acc": 95.8, "miou": 53.0},
    },
}

# Table 5 — Stanford2D3D depth, rank 8
TABLE5_DEPTH_RANK8 = {
    "PanoFormer": {"params_m": 14.5, "flops_g": 44.7, "mae": 0.167, "mre": 0.072, "d1": 93.7},
    "Elite360D": {"params_m": 14.7, "flops_g": 51.4, "mae": 0.181, "mre": 0.077, "d1": 93.2},
    "OURS": {"params_m": 14.9, "flops_g": 52.7, "nodes_k": 655, "mae": 0.147, "mre": 0.065, "d1": 94.0},
}

# Table 6 — positional encoding ablation (depth S2D3D)
TABLE6_POS_ABLATION = {
    "No_Pos": {"abs": False, "rel": False, "mae": 0.326, "mre": 0.094},
    "No_Rel": {"abs": True, "rel": False, "mae": 0.251, "mre": 0.091},
    "No_Abs": {"abs": False, "rel": True, "mae": 0.218, "mre": 0.088},
    "With_Pos": {"abs": True, "rel": True, "mae": 0.189, "mre": 0.077},
}

# Table 7 — Chead / Cwin ablation
TABLE7_SCALE_ABLATION = {
    "Base": {"c_head": 1, "c_win": 1, "mae": 0.189, "mre": 0.077},
    "No_window": {"c_head": 1, "c_win": 0, "mae": 0.412, "mre": 0.122},
    "Med_window": {"c_head": 1, "c_win": 2, "mae": 0.175, "mre": 0.073},
    "Conf_1": {"c_head": 2, "c_win": 2, "mae": 0.165, "mre": 0.071},
    "Large_head": {"c_head": 4, "c_win": 1, "mae": 0.162, "mre": 0.067},
}

TABLE3_CONFIGS = {
    "rank7_hex": {"rank": 7, "node_type": "hex", "c_head": 2, "c_win": 2, "nodes_k": 164, "params_m": 14.9},
    "rank8_hex": {"rank": 8, "node_type": "hex", "c_head": 2, "c_win": 2, "nodes_k": 655, "params_m": 14.9},
}


def beats_panoformer_depth_s2d3d() -> bool:
    o, p = TABLE4_RANK7["OURS"]["s2d3d"], TABLE4_RANK7["PanoFormer"]["s2d3d"]
    return o["mae"] < p["mae"] and o["d1"] > p["d1"]


def beats_elite360d_seg_s2d3d() -> bool:
    o, e = TABLE4_RANK7["OURS"]["s2d3d_seg"], TABLE4_RANK7["Elite360D"]["s2d3d_seg"]
    return o["miou"] > e["miou"]


def rank8_beats_panoformer_mae() -> bool:
    return TABLE5_DEPTH_RANK8["OURS"]["mae"] < TABLE5_DEPTH_RANK8["PanoFormer"]["mae"]


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "arxiv": PAPER_ARXIV,
        "title": PAPER_TITLE,
        "url": PAPER_URL,
        "table4_rank7": TABLE4_RANK7,
        "table5_depth_rank8": TABLE5_DEPTH_RANK8,
        "table6_pos_ablation": TABLE6_POS_ABLATION,
        "table7_scale_ablation": TABLE7_SCALE_ABLATION,
        "table3_configs": TABLE3_CONFIGS,
    }
