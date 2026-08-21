"""Reference metrics from Wang et al. (arXiv:2605.13169)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.panoworld.config import (
    CORPUS_SIZE,
    INSTRUCTION_CANONICAL,
    PAPER_ARXIV,
    PAPER_TITLE,
    PAPER_URL,
    PROJECT_URL,
)

# Table 2 — PanoSpace-Bench
TABLE2_PANOSPACE: dict[str, float] = {
    "overall": 56.5,
    "abs_dir": 93.7,
    "bfov_miou": 73.3,
    "rel_dir": 42.6,
    "cam_rot": 52.4,
    "obj_reori": 47.2,
    "spherical_relation_avg": 47.4,
    "dist": 59.6,
    "rel_3d": 40.6,
    "spatial_3d_avg": 49.8,
    "seam": 65.5,
    "Qwen3.5-9B_overall": 30.8,
}

# Table 3 — H*Bench
TABLE3_HSTAR: dict[str, Any] = {
    "zero_shot_overall": 56.1,
    "zero_shot_HOS": 61.8,
    "zero_shot_HPS": 47.5,
    "sft_overall": 70.1,
    "sft_HOS": 73.1,
    "sft_HPS": 64.2,
    "HVS-3B_perspective": 38.4,
}

# Table 4 — R2R-CE Val-Unseen (PanoWorld-VLN)
TABLE4_R2R_CE: dict[str, float] = {
    "NE": 4.98,
    "OSR": 59.3,
    "SR": 54.3,
    "SPL": 52.1,
    "GridMM_SR": 49.0,
    "StreamVLN_SR": 50.2,
}

# Table 5 — ability-oriented training ablation (full row)
TABLE5_ABILITY_ABLATION: list[dict[str, Any]] = [
    {"semantic": True, "spherical": False, "reframe": False, "depth3d": False, "overall": 24.0},
    {"semantic": False, "spherical": True, "reframe": False, "depth3d": False, "overall": 59.2},
    {"semantic": False, "spherical": False, "reframe": True, "depth3d": False, "overall": 36.4},
    {"semantic": False, "spherical": False, "reframe": False, "depth3d": True, "overall": 32.0},
    {
        "semantic": True,
        "spherical": True,
        "reframe": True,
        "depth3d": True,
        "overall": 68.8,
        "bfov_miou": 66.7,
    },
]

# Table 6 — metadata verification
TABLE6_VERIFICATION: list[dict[str, Any]] = [
    {"det_verif": False, "sem_verif": False, "overall": 38.8},
    {"det_verif": True, "sem_verif": False, "overall": 46.4},
    {"det_verif": False, "sem_verif": True, "overall": 48.0},
    {"det_verif": True, "sem_verif": True, "overall": 55.1},
]

# Table 7 — architecture (Cross-Attn Patch = full SSCA)
TABLE7_ARCHITECTURE: list[dict[str, Any]] = [
    {"method": "Qwen3.5", "position": "-", "overall": 48.40},
    {"method": "Cross-Attn", "position": "Patch", "overall": 55.10, "bfov_miou": 72.60},
]

# Table 8 — trainable scope (full FT row)
TABLE8_TRAIN_SCOPE: dict[str, float] = {
    "LLM_only_abs_dir": 84.80,
    "full_abs_dir": 92.80,
    "full_bfov_miou": 72.60,
}

# Table 1 — resource comparison (corpus)
TABLE1_RESOURCES: dict[str, Any] = {
    "PanoWorld_panoramas": CORPUS_SIZE,
    "instruction_canonical": INSTRUCTION_CANONICAL,
    "depth": True,
    "entity_metadata": True,
    "verified_graph": True,
    "Dense360_panoramas": 160_000,
    "PanoVQA_panoramas": 44_600,
}

TRAINING = {
    "base_model": "Qwen3.5-VL",
    "gpus": 8,
    "gpu": "A100",
    "optimizer": "AdamW",
    "lr": 1e-6,
    "batch_size": 2,
    "grad_accum": 4,
    "epochs": 1,
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": {"arxiv": PAPER_ARXIV, "title": PAPER_TITLE, "url": PAPER_URL, "project": PROJECT_URL},
        "table1_resources": TABLE1_RESOURCES,
        "table2_panospace": TABLE2_PANOSPACE,
        "table3_hstar": TABLE3_HSTAR,
        "table4_r2r_ce": TABLE4_R2R_CE,
        "table5_ability_ablation": TABLE5_ABILITY_ABLATION,
        "table6_verification": TABLE6_VERIFICATION,
        "table7_architecture": TABLE7_ARCHITECTURE,
        "table8_train_scope": TABLE8_TRAIN_SCOPE,
        "training": TRAINING,
    }
