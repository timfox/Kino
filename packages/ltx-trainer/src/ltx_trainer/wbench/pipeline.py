"""WBENCH framework card, paper tables, and smoke demos (Ying et al., arXiv:2605.25874)."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.wbench.config import WBENCHConfig
from ltx_trainer.wbench.metrics import (
    batch_adjacent_cosine,
    causal_fidelity_case_score,
    event_editing_turn_score,
    hpsv3_norm,
    perspective_switching_turn_score,
    subject_action_turn_score,
    temporal_flickering_score,
)
from ltx_trainer.wbench.navigation import (
    action_to_text,
    build_orbit_trajectory,
    build_translation_trajectory,
    nav_score_from_trajectories,
)


def framework_card(cfg: WBENCHConfig | None = None) -> dict[str, Any]:
    cfg = cfg or WBENCHConfig()
    return {
        "name": "WBENCH",
        "paper": cfg.paper_arxiv,
        "title": "A Comprehensive Multi-turn Benchmark for Interactive Video World Model Evaluation",
        "authors": "Ying, Hu, Ren, Li, Chen, Wang, Cao, Cai, Ding (Fudan + Meituan Longcat)",
        "repo": cfg.repo_url,
        "cases": cfg.num_cases,
        "turns": cfg.num_turns,
        "nav_subset_cases": cfg.nav_cases,
        "models_evaluated": cfg.num_evaluated_models,
        "sub_metrics": cfg.num_sub_metrics,
        "dimensions": list(cfg.dimensions),
        "interaction_types": list(cfg.interaction_types),
        "control_paradigms": ("text", "6dof_pose", "discrete_action"),
        "vlm_judge": cfg.vlm_model,
        "finding": "No single model dominates all five dimensions",
    }


def sub_metric_index() -> list[dict[str, str]]:
    """Table 8 quick reference — 22 sub-metrics across five dimensions."""
    return [
        {"dim": "video_quality", "id": "V.1", "name": "Aesthetic Quality", "tool": "CLIP+LAION"},
        {"dim": "video_quality", "id": "V.2", "name": "Imaging Quality", "tool": "MUSIQ"},
        {"dim": "video_quality", "id": "V.3", "name": "Temporal Flickering", "tool": "pixel MAE"},
        {"dim": "video_quality", "id": "V.4", "name": "Dynamic Degree", "tool": "RAFT flow"},
        {"dim": "video_quality", "id": "V.5", "name": "Motion Smoothness", "tool": "AMT-S"},
        {"dim": "video_quality", "id": "V.6", "name": "HPSv3-Norm", "tool": "HPSv3"},
        {"dim": "setting_adherence", "id": "S.1", "name": "Scene Adherence", "tool": "VLM"},
        {"dim": "setting_adherence", "id": "S.2", "name": "Subject Adherence", "tool": "VLM"},
        {"dim": "interaction_adherence", "id": "I.1", "name": "NavScore", "tool": "MegaSaM"},
        {"dim": "interaction_adherence", "id": "I.2", "name": "Event Editing Adherence", "tool": "VLM"},
        {"dim": "interaction_adherence", "id": "I.3", "name": "Subject Action Adherence", "tool": "VLM"},
        {"dim": "interaction_adherence", "id": "I.4", "name": "Perspective Switching", "tool": "VLM"},
        {"dim": "consistency", "id": "C.1", "name": "Subject Consistency", "tool": "SAM2+DINO+CLIP"},
        {"dim": "consistency", "id": "C.2", "name": "Background Consistency", "tool": "CLIP"},
        {"dim": "consistency", "id": "C.3", "name": "Spatial Consistency", "tool": "MegaSaM+DreamSim"},
        {"dim": "consistency", "id": "C.4", "name": "Gated Spatial Consistency", "tool": "MegaSaM+DreamSim"},
        {"dim": "consistency", "id": "C.5", "name": "Segment Continuity", "tool": "TransNetV2"},
        {"dim": "consistency", "id": "C.6", "name": "Perspective Consistency", "tool": "SAM2"},
        {"dim": "consistency", "id": "C.7", "name": "Geometric Consistency", "tool": "Depth Anything 3"},
        {"dim": "consistency", "id": "C.8", "name": "Photometric Consistency", "tool": "Depth Anything 3"},
        {"dim": "physical", "id": "P.1", "name": "Causal Fidelity", "tool": "VLM two-track"},
        {"dim": "physical", "id": "P.2", "name": "Visual Plausibility", "tool": "Qwen3-VL ft."},
    ]


def dataset_composition() -> dict[str, Any]:
    """Fig. 2 — dataset composition (Sec. 3.2)."""
    return {
        "cases": 289,
        "turns": 1058,
        "perspective": {"fpp_pct": 62, "tpp_pct": 38},
        "interaction_share_pct": {
            "navigation": 57,
            "subject_action": 20,
            "event_editing": 17,
            "perspective_switching": 6,
        },
        "avg_turns_per_case": 3.7,
        "scene_pct": {
            "nature": 31,
            "urban": 21,
            "indoor": 17,
            "workspace": 13,
            "fantasy": 10,
            "sports": 8,
        },
        "subject_pct": {"human": 64, "animal": 9, "robot": 9, "vehicle": 7, "other": 10},
        "photorealistic_pct": 52,
        "perspective_switch_turns": 61,
    }


def benchmark_comparison() -> dict[str, dict[str, bool | int]]:
    """Table 1 — WBENCH vs representative benchmarks."""
    return {
        "VBench": {"cases": 946, "fpp": False, "tpp": False, "multi_turn": False, "physics": False},
        "WorldMark": {"cases": 500, "fpp": True, "tpp": True, "multi_turn": False, "physics": False},
        "Omni-WorldBench": {"cases": 1068, "fpp": True, "tpp": False, "multi_turn": True, "physics": True},
        "WBENCH": {
            "cases": 289,
            "fpp": True,
            "tpp": True,
            "multi_turn": True,
            "physics": True,
            "turns": 1058,
        },
    }


def table_navigation_split() -> dict[str, dict[str, float]]:
    """Table 2 — dimension averages on 158-case navigation split (scores ∈ [0, 100])."""
    return {
        "Seedance 1.5": {
            "video_quality": 82.1,
            "setting_adherence": 82.9,
            "navigation": 68.0,
            "consistency": 81.3,
            "physical": 68.3,
        },
        "Wan 2.7": {
            "video_quality": 81.5,
            "setting_adherence": 91.4,
            "navigation": 66.0,
            "consistency": 81.6,
            "physical": 71.8,
        },
        "Kling 3.0": {
            "video_quality": 81.4,
            "setting_adherence": 91.0,
            "navigation": 70.3,
            "consistency": 83.7,
            "physical": 69.3,
        },
        "YUME 1.5": {
            "video_quality": 77.6,
            "setting_adherence": 72.4,
            "navigation": 72.0,
            "consistency": 80.1,
            "physical": 65.2,
        },
        "LingBot-World": {
            "video_quality": 78.9,
            "setting_adherence": 72.6,
            "navigation": 79.8,
            "consistency": 89.9,
            "physical": 71.2,
        },
        "HY-World 1.5": {
            "video_quality": 78.1,
            "setting_adherence": 72.2,
            "navigation": 87.5,
            "consistency": 86.9,
            "physical": 66.3,
        },
        "Happy Oyster": {
            "video_quality": 77.3,
            "setting_adherence": 74.2,
            "navigation": 85.1,
            "consistency": 84.3,
            "physical": 63.5,
        },
        "Matrix-Game 3.0": {
            "video_quality": 75.5,
            "setting_adherence": 63.7,
            "navigation": 83.5,
            "consistency": 74.5,
            "physical": 59.4,
        },
        "Genie 3": {
            "video_quality": 75.2,
            "setting_adherence": 72.5,
            "navigation": 73.3,
            "consistency": 82.6,
            "physical": 65.7,
        },
        "text-driven average": {
            "video_quality": 77.7,
            "setting_adherence": 81.6,
            "navigation": 67.6,
            "consistency": 83.1,
            "physical": 67.0,
        },
        "camera-controlled average": {
            "video_quality": 73.6,
            "navigation": 76.0,
            "consistency": 85.0,
            "physical": 64.2,
        },
        "action-conditioned average": {
            "video_quality": 75.3,
            "navigation": 77.7,
            "consistency": 76.5,
            "physical": 61.7,
        },
    }


def table_full_text_driven() -> dict[str, dict[str, float]]:
    """Table 10 — full 289-case text-driven interaction averages (Sec. D.1)."""
    return {
        "Kling 3.0": {
            "video_quality": 80.0,
            "setting_adherence": 91.0,
            "interaction_avg": 73.1,
            "event_editing": 81.4,
            "subject_action": 85.6,
            "perspective_switching": 55.0,
            "consistency": 83.9,
            "physical": 69.2,
        },
        "Wan 2.7": {
            "video_quality": 81.0,
            "setting_adherence": 91.5,
            "interaction_avg": 72.1,
            "event_editing": 84.0,
            "subject_action": 83.4,
            "perspective_switching": 55.0,
            "consistency": 75.8,
            "physical": 71.6,
        },
        "YUME 1.5": {
            "interaction_avg": 48.4,
            "perspective_switching": 16.7,
        },
        "perspective_switching_avg_all": 30.7,
        "navigation_turn_decay_t4": -33,
        "event_editing_turn_decay_t4": -13,
        "subject_action_turn_decay_t4": -9,
        "perspective_switch_turn_delta_t4": 2,
    }


def human_preference_alignment() -> dict[str, float]:
    """Fig. 5 — Spearman ρ between human win rate and automated scores (Sec. 5.4)."""
    return {
        "navigation": 0.94,
        "event_editing": 1.00,
        "subject_action": 1.00,
        "perspective_switching": 1.00,
        "spatial_consistency": 1.00,
        "physical_average": 0.94,
        "scene_adherence": 0.94,
        "subject_adherence": 0.94,
        "perspective_consistency": 0.94,
        "subject_consistency": 0.94,
        "min_rho_reported": 0.94,
    }


def training_step_demo(cfg: WBENCHConfig | None = None) -> dict[str, float]:
    """Smoke: NavScore, VLM checklist scores, and rescaling helpers."""
    cfg = cfg or WBENCHConfig()
    gt = build_translation_trajectory("W", length=1.0, num_points=cfg.arc_length_samples)
    pred = gt + torch.randn_like(gt) * 0.02
    nav = nav_score_from_trajectories(pred, gt, num_samples=cfg.arc_length_samples) * 100.0

    ee = event_editing_turn_score(
        static_scene=False,
        event_occurs=True,
        event_complete=True,
        detail_accurate=True,
        no_anomaly=True,
    )
    sa = subject_action_turn_score(
        subject_idle=False,
        action_occurs=True,
        action_complete=True,
        detail_accurate=True,
        natural_motion=True,
    )
    ps = perspective_switching_turn_score(True, True, True)
    causal = causal_fidelity_case_score(2.5, [2.0, 3.0, 2.5])

    feats = torch.randn(8, 16)
    subj_cons = batch_adjacent_cosine(feats)

    return {
        "nav_score": nav,
        "event_editing": ee,
        "subject_action": sa,
        "perspective_switch": ps,
        "causal_fidelity": causal,
        "hpsv3_norm": hpsv3_norm(7.2, cfg.hpsv3_p1, cfg.hpsv3_p99),
        "flickering": temporal_flickering_score(8.0),
        "subject_consistency_smoke": subj_cons,
        "prompt_w_fpp": 1.0,  # placeholder length for JSON stability
    }


def evaluation_demo(cfg: WBENCHConfig | None = None) -> dict[str, Any]:
    """Smoke: paper tables, cross-paradigm nav leaders, human alignment."""
    cfg = cfg or WBENCHConfig()
    step = training_step_demo(cfg)
    nav_tab = table_navigation_split()
    comp = dataset_composition()

    hy_nav = nav_tab["HY-World 1.5"]["navigation"]
    kling_nav = nav_tab["Kling 3.0"]["navigation"]
    ling_cons = nav_tab["LingBot-World"]["consistency"]

    return {
        **step,
        "cases": float(cfg.num_cases),
        "turns": float(cfg.num_turns),
        "fpp_pct": float(comp["perspective"]["fpp_pct"]),
        "nav_share_pct": float(comp["interaction_share_pct"]["navigation"]),
        "hy_world_nav_leads": hy_nav > kling_nav,
        "lingbot_consistency_leads": ling_cons >= 89.0,
        "perspective_switch_avg": table_full_text_driven()["perspective_switching_avg_all"],
        "human_alignment_min_rho": human_preference_alignment()["min_rho_reported"],
        "wbench_vs_vbench_multi_turn": benchmark_comparison()["WBENCH"]["multi_turn"],
        "action_to_text_w": action_to_text("W", "fpp"),
        "sub_metric_count": float(len(sub_metric_index())),
    }
