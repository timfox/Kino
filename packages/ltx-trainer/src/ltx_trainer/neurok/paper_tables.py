"""Paper Tables 1–2 and baseline registry (arXiv:2605.30347)."""

from __future__ import annotations

from typing import Any


def table_i_inverse_kinematics() -> list[dict[str, Any]]:
    """Table 1 — PartNet-Mobility inverse kinematics (lower Chamfer better, higher IoU better)."""
    return [
        {"method": "NeuralDeformationGraphs", "chamfer_l1": 0.670, "chamfer_l2": 0.724, "iou": 0.289},
        {"method": "SINGAPO", "chamfer_l1": 0.313, "chamfer_l2": 0.200, "iou": 0.091},
        {"method": "FreeArt3D", "chamfer_l1": 0.169, "chamfer_l2": 0.139, "iou": 0.354},
        {"method": "CANOR", "chamfer_l1": 0.082, "chamfer_l2": 0.067, "iou": 0.568},
        {"method": "KeyPointDeformer", "chamfer_l1": 0.067, "chamfer_l2": 0.067, "iou": 0.570},
        {"method": "NEUROK (ours)", "chamfer_l1": 0.028, "chamfer_l2": 0.028, "iou": 0.764},
        {"method": "NEUROK w/o Model Reduction", "chamfer_l1": 0.045, "chamfer_l2": 0.059, "iou": 0.711},
        {"method": "NEUROK w/o Data Augmentation", "chamfer_l1": 0.036, "chamfer_l2": 0.041, "iou": 0.724},
        {"method": "NEUROK w/o Dual-Quaternion", "chamfer_l1": 0.033, "chamfer_l2": 0.037, "iou": 0.728},
    ]


def table_ii_generative_4d() -> list[dict[str, Any]]:
    """Table 2 — physically-inspired 4D generation (user study + VBench + WorldScore)."""
    return [
        {
            "method": "PhysDreamer",
            "user_alignment_pct": 5.95,
            "user_realism_pct": 5.36,
            "vbench_aq": 0.362,
            "vbench_dd": 0.500,
            "vbench_iq": 48.432,
            "worldscore_clip": 0.716,
            "worldscore_mm": 0.783,
        },
        {
            "method": "OmniPhysGS",
            "user_alignment_pct": 1.67,
            "user_realism_pct": 0.48,
            "vbench_aq": 0.380,
            "vbench_dd": 0.625,
            "vbench_iq": 48.937,
            "worldscore_clip": 0.690,
            "worldscore_mm": 0.544,
        },
        {
            "method": "Pixie",
            "user_alignment_pct": 5.12,
            "user_realism_pct": 4.17,
            "vbench_aq": 0.392,
            "vbench_dd": 0.625,
            "vbench_iq": 46.177,
            "worldscore_clip": 0.659,
            "worldscore_mm": 0.857,
        },
        {
            "method": "AnimateAnyMesh",
            "user_alignment_pct": 5.83,
            "user_realism_pct": 6.67,
            "vbench_aq": 0.450,
            "vbench_dd": 0.625,
            "vbench_iq": 48.370,
            "worldscore_clip": 0.730,
            "worldscore_mm": 0.889,
        },
        {
            "method": "NEUROK (ours)",
            "user_alignment_pct": 81.43,
            "user_realism_pct": 83.33,
            "vbench_aq": 0.483,
            "vbench_dd": 0.750,
            "vbench_iq": 51.100,
            "worldscore_clip": 0.761,
            "worldscore_mm": 2.343,
        },
    ]


def ablation_rows() -> dict[str, dict[str, float]]:
    """Table 1 ablation lines keyed by variant name."""
    rows = table_i_inverse_kinematics()
    return {r["method"]: r for r in rows if "w/" in r["method"] or r["method"] == "NEUROK (ours)"}


def user_study_headline() -> dict[str, float]:
    ours = next(r for r in table_ii_generative_4d() if r["method"] == "NEUROK (ours)")
    return {
        "alignment_pct": float(ours["user_alignment_pct"]),
        "realism_pct": float(ours["user_realism_pct"]),
        "num_users": 105.0,
    }
