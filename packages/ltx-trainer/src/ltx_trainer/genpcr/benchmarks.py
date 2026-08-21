"""Reference metrics (Jiang et al., arXiv:2512.09407)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.genpcr.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL

# Table 2 — ScanNet, Generative FCGF SD (×4 rotation @45° etc.)
SCANNET_GENERATIVE_FCGF_SD = {
    "rot_45_acc": 98.1,
    "rot_mean_err": 4.5,
    "trans_25_acc": 82.7,
    "chamfer_10_acc": 94.6,
    "chamfer_mean_mm": 37.7,
}

# Table 3 — 3DMatch Generative FCGF SD
THREEDMATCH_GENERATIVE_FCGF_SD = {
    "rot_45_acc": 98.1,
    "rot_mean_err": 4.5,
    "trans_25_acc": 93.1,
    "chamfer_10_acc": 94.6,
}

# Table 4 — Dur360BEV ≥5m, Generative variants
DUR360_GENERATIVE = {
    "FPFH_SD_RR": 88.6,
    "Predator_SD_IR": 33.3,
    "Predator_SD_RR": 98.4,
    "GeoTrans_SD_IR": 63.8,
}

# Table 5 — 3DMatch ablation Generative FCGF SD
THREEDMATCH_ABLATION_FCGF_SD = {
    "geo_only_rot_45": 97.8,
    "geo_tex_rot_45": 98.1,
    "zero_shot_rot_45": 97.3,
    "finetune_rot_45": 98.1,
    "finetune_1k_rot_45": 98.1,
    "finetune_3k_rot_45": 98.1,
    "finetune_5k_rot_45": 98.0,
    "fusion_weight_default": 0.5,
    "color_dim_default": 64,
}

THREEDMATCH_GENERATIVE_COLORPCR = {
    "rot_45_acc": 93.2,
    "trans_25_acc": 86.7,
    "chamfer_10_acc": 88.2,
}

# ScanNet Table 2 — Generative FCGF SD improvement row
SCANNET_IMPROVEMENT_ROT45 = 6.9


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "url": PAPER_URL,
        "scannet_generative_fcgf_sd": SCANNET_GENERATIVE_FCGF_SD,
        "scannet_improvement_rot45": SCANNET_IMPROVEMENT_ROT45,
        "threedmatch_generative_fcgf_sd": THREEDMATCH_GENERATIVE_FCGF_SD,
        "dur360_generative": DUR360_GENERATIVE,
        "threedmatch_ablation_fcgf_sd": THREEDMATCH_ABLATION_FCGF_SD,
        "threedmatch_generative_colorpcr": THREEDMATCH_GENERATIVE_COLORPCR,
    }
