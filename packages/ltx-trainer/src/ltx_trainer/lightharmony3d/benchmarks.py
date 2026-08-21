"""Reference metrics from LightHarmony3D paper (arXiv:2603.29209)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.lightharmony3d.config import GENENV_TRAINING_HDRIS
from ltx_trainer.lightharmony3d.metrics import (
    TABLE1_LH3D_KU,
    TABLE2_VQA_MIPNERF360,
    TABLE3_LH3D_BLENDER,
    TABLE4_ABLATION,
)

PAPER_ARXIV = "2603.29209"
PAPER_TITLE = "LightHarmony3D: Harmonizing Illumination and Shadows for Object Insertion in 3D Gaussian Splatting"

TRAINING_DEFAULTS = {
    "reconstruction": "MILo",
    "gen_env_backbone": "Flux.1 Kontext + LoRA",
    "hdr_training_panoramas": GENENV_TRAINING_HDRIS,
    "pbr_engine": "Blender Cycles (stub)",
    "inference_ode_steps": 40,
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_lh3d_ku": TABLE1_LH3D_KU,
        "table2_vqa_mipnerf360": TABLE2_VQA_MIPNERF360,
        "table3_lh3d_blender": TABLE3_LH3D_BLENDER,
        "table4_ablation": TABLE4_ABLATION,
        "training_defaults": TRAINING_DEFAULTS,
    }
