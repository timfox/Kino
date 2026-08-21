"""Reference tables from Zhang et al. (arXiv:2606.13652)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.world_tracing.config import (
    DYNAMIC_CLIPS,
    OBJECT_CORPUS_ASSETS,
    OBJECT_RENDER_VIEWS,
    PAPER_ARXIV,
    PAPER_TITLE,
    PAPER_URL,
    PROJECT_URL,
)

# Table 1 — object visible surface + full geometry
TABLE1_OBJECT_VISIBLE: dict[str, dict[str, float]] = {
    "DA3": {"mae": 0.0703, "rmse": 0.0920, "abs_rel": 0.0384, "d125": 0.9973},
    "LaRI": {"mae": 0.0366, "rmse": 0.0506, "abs_rel": 0.0198, "d125": 0.9992},
    "Pi3X": {"mae": 0.0317, "rmse": 0.0440, "abs_rel": 0.0172, "d125": 0.9994},
    "MoGe-2": {"mae": 0.0261, "rmse": 0.0368, "abs_rel": 0.0141, "d125": 0.9995},
    "VGGT": {"mae": 0.0257, "rmse": 0.0370, "abs_rel": 0.0138, "d125": 0.9995},
    "WT-O": {"mae": 0.0149, "rmse": 0.0243, "abs_rel": 0.0079, "d125": 0.9996},
}

TABLE1_OBJECT_FULL: dict[str, dict[str, float | str]] = {
    "TRELLIS.2": {"l1": 0.0566, "l2": 0.00717, "f001": 0.204, "f005": 0.598, "output": "Mesh"},
    "SAM 3D": {"l1": 0.0475, "l2": 0.00501, "f001": 0.203, "f005": 0.675, "output": "3DGS"},
    "LaS-Comp": {"l1": 0.0477, "l2": 0.00739, "f001": 0.225, "f005": 0.677, "output": "Mesh"},
    "ReconViaGen": {"l1": 0.0478, "l2": 0.00528, "f001": 0.228, "f005": 0.677, "output": "Mesh"},
    "WT-O*": {"l1": 0.0326, "l2": 0.00530, "f001": 0.321, "f005": 0.808, "output": "Mesh"},
    "WT-O": {"l1": 0.0213, "l2": 0.00194, "f001": 0.549, "f005": 0.898, "output": "PC"},
}

# Table 2 — scene geometry (3D-FRONT held-out excerpt)
TABLE2_SCENE_3DFRONT: dict[str, dict[str, float]] = {
    "VGGT": {"mae": 0.0393, "rmse": 0.0568, "l0_cd_l1": 0.0345, "all_l_cd_l1": float("nan")},
    "MoGe-2": {"mae": 0.0248, "rmse": 0.0373, "l0_cd_l1": 0.0213, "all_l_cd_l1": float("nan")},
    "LaRI-scene": {"mae": 0.0319, "rmse": 0.0483, "l0_cd_l1": 0.0268, "all_l_cd_l1": 0.0575},
    "WT-S*": {"mae": 0.0106, "rmse": 0.0220, "l0_cd_l1": 0.0097, "all_l_cd_l1": 0.0224},
    "WT-S": {"mae": 0.0102, "rmse": 0.0215, "l0_cd_l1": 0.0093, "all_l_cd_l1": 0.0216},
}

# Table 3 — dynamic geometry CD-L2
TABLE3_DYNAMIC: dict[str, dict[str, float]] = {
    "GVFD": {"actionbench": 0.0879, "truebone": 0.0166, "obj_val": 0.0248, "mean": 0.0385},
    "SS4D": {"actionbench": 0.0882, "truebone": 0.0145, "obj_val": 0.0249, "mean": 0.0381},
    "AM": {"actionbench": 0.0243, "truebone": 0.0120, "obj_val": 0.0142, "mean": 0.0162},
    "WT-D": {"actionbench": 0.0291, "truebone": 0.0063, "obj_val": 0.0034, "mean": 0.0105},
}

# Table 4 — timestep sampling ablation
TABLE4_TIMESTEP: dict[str, dict[str, float]] = {
    "Plat. logit-normal": {"l0": 0.021, "l1_l5": 0.033, "all": 0.031},
    "Logit-normal": {"l0": 0.023, "l1_l5": 0.028, "all": 0.027},
    "Mixture": {"l0": 0.020, "l1_l5": 0.025, "all": 0.024},
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "arxiv": PAPER_ARXIV,
        "title": PAPER_TITLE,
        "paper_url": PAPER_URL,
        "project_url": PROJECT_URL,
        "table1_object_visible": TABLE1_OBJECT_VISIBLE,
        "table1_object_full": TABLE1_OBJECT_FULL,
        "table2_scene_3dfront": TABLE2_SCENE_3DFRONT,
        "table3_dynamic": TABLE3_DYNAMIC,
        "table4_timestep": TABLE4_TIMESTEP,
        "training": {
            "object_assets": OBJECT_CORPUS_ASSETS,
            "object_render_views": OBJECT_RENDER_VIEWS,
            "dynamic_clips": DYNAMIC_CLIPS,
            "layers": 6,
            "resolution": 504,
            "ode_steps": 20,
        },
    }
