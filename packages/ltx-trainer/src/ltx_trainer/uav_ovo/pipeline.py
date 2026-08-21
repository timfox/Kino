"""UAV-OVO framework card, paper tables, and smoke demos (arXiv:2605.25615)."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.uav_ovo.config import UAVOVOConfig
from ltx_trainer.uav_ovo.later import (
    OnlineTargetCenter,
    build_lora_subspace,
    later_recenter,
    orthogonal_projector,
)
from ltx_trainer.uav_ovo.metrics import harmonic_mean, performance_drop
from ltx_trainer.uav_ovo.view_score import (
    ViewSplit,
    angle_optical_axis_to_normal,
    assign_split,
    pitch_offset_deg,
    video_view_score,
)


def framework_card(cfg: UAVOVOConfig | None = None) -> dict[str, Any]:
    cfg = cfg or UAVOVOConfig()
    return {
        "name": "UAV-OVO",
        "paper": cfg.paper_arxiv,
        "task": "155-class UAV action recognition, out-of-viewpoint generalization",
        "splits": {
            "train_id": f"0–{cfg.train_view_max}°",
            "isolation": f"{cfg.isolation_view_min}–{cfg.isolation_view_max}°",
            "ood": f">{cfg.ood_view_min}° (+{cfg.ood_topup_videos} top-up)",
        },
        "total_videos": cfg.total_videos,
        "method": "LATER (LoRA-anchored test-time re-centering)",
        "backbone": cfg.backbone,
    }


def table_split_statistics() -> dict[str, dict[str, int | float]]:
    """Table 2 — UAV-OVO split statistics."""
    return {
        "Train": {"view_range": "0–30°", "videos": 13872, "classes": 155},
        "ID test": {"view_range": "0–30°", "videos": 3100, "classes": 155},
        "Isolation band": {"view_range": "30–40°", "videos": 3275, "classes": 155},
        "OOD test": {"view_range": ">40°", "videos": 3100, "classes": 155},
        "Total": {"view_range": "—", "videos": 23347, "classes": 155},
    }


def table_main_results() -> dict[str, dict[str, float]]:
    """Table 3 — main UAV-OVO results (MViTv2-B family, percent accuracies)."""
    return {
        "ViT-S": {"acc_id": 47.71, "acc_ood": 15.58, "pd": 0.67, "h": 23.49},
        "X3D-M": {"acc_id": 68.45, "acc_ood": 25.45, "pd": 0.63, "h": 37.10},
        "MViTv2-B": {"acc_id": 66.45, "acc_ood": 33.94, "pd": 0.49, "h": 44.93},
        "NEO": {"acc_id": 66.71, "acc_ood": 34.58, "pd": 0.48, "h": 45.55},
        "LATER": {"acc_id": 67.06, "acc_ood": 35.55, "pd": 0.47, "h": 46.47},
    }


def table_lora_rank_ablation() -> dict[str, dict[str, float]]:
    """Table 4 — LoRA rank sweep."""
    return {
        "r=2": {"acc_id": 60.55, "acc_ood": 37.23, "pd": 0.39, "h": 46.11},
        "r=4": {"acc_id": 62.42, "acc_ood": 37.97, "pd": 0.39, "h": 47.22},
        "r=8": {"acc_id": 65.81, "acc_ood": 36.58, "pd": 0.44, "h": 47.02},
        "r=16": {"acc_id": 67.06, "acc_ood": 35.55, "pd": 0.47, "h": 46.47},
    }


def table_alpha_ablation() -> dict[str, dict[str, float]]:
    """Table 5 — re-centering strength α."""
    return {
        "α=0.75": {"acc_id": 67.84, "acc_ood": 35.19, "pd": 0.48, "h": 46.34},
        "α=1.00": {"acc_id": 67.06, "acc_ood": 35.55, "pd": 0.47, "h": 46.47},
        "α=1.25": {"acc_id": 66.35, "acc_ood": 35.61, "pd": 0.46, "h": 46.35},
        "α=1.50": {"acc_id": 65.71, "acc_ood": 36.13, "pd": 0.45, "h": 46.62},
    }


def table_projection_ablation() -> dict[str, dict[str, float]]:
    """Table 6 — global vs LoRA-anchored correction."""
    return {
        "Global Correction": {"acc_id": 63.29, "acc_ood": 36.23, "pd": 0.43, "h": 46.08},
        "LATER": {"acc_id": 67.06, "acc_ood": 35.55, "pd": 0.47, "h": 46.47},
    }


def training_step_demo(cfg: UAVOVOConfig | None = None) -> dict[str, float]:
    """Smoke: view score, split tag, LATER re-centering."""
    cfg = cfg or UAVOVOConfig()
    torch.manual_seed(15)
    d = cfg.feature_dim

    o = torch.tensor([0.2, -0.9, 0.4])
    n = torch.tensor([0.0, -1.0, 0.0])
    theta = angle_optical_axis_to_normal(o.unsqueeze(0), n.unsqueeze(0))
    scores = pitch_offset_deg(theta)
    v_score = float(video_view_score(scores))
    split = assign_split(v_score, cfg)

    b1 = torch.randn(d, cfg.lora_rank)
    b2 = torch.randn(d, cfg.lora_rank)
    u = build_lora_subspace([b1, b2], d, cfg.svd_keep_ratio)
    p_perp = orthogonal_projector(u, d)

    mu_s = torch.randn(d)
    center = OnlineTargetCenter(cfg.queue_size)
    for _ in range(5):
        center.push(torch.randn(d))
    mu_t = center.mean()
    h = torch.randn(d)
    h_tilde = later_recenter(h, mu_s, mu_t, p_perp, alpha=cfg.recenter_alpha)

    acc_id, acc_ood = 67.06, 35.55
    return {
        "view_score_deg": v_score,
        "split_is_low_depression": float(split in (ViewSplit.TRAIN, ViewSplit.ID_TEST) if split else 0),
        "lora_subspace_dim": float(u.shape[1]),
        "recenter_norm": float((h - h_tilde).norm()),
        "pd": performance_drop(acc_id, acc_ood),
        "h": harmonic_mean(acc_id, acc_ood),
    }


def evaluation_demo(cfg: UAVOVOConfig | None = None) -> dict[str, Any]:
    """Smoke: paper table ordering and benchmark claims."""
    cfg = cfg or UAVOVOConfig()
    step = training_step_demo(cfg)
    t3 = table_main_results()
    t6 = table_projection_ablation()

    baselines = ["MViTv2-B", "NEO", "LATER"]
    ood_accs = {m: t3[m]["acc_ood"] for m in baselines}
    h_scores = {m: t3[m]["h"] for m in baselines}

    return {
        **step,
        "later_best_ood_among_listed": t3["LATER"]["acc_ood"] == max(ood_accs.values()),
        "later_best_h_among_listed": t3["LATER"]["h"] == max(h_scores.values()),
        "later_beats_mvitv2_ood": t3["LATER"]["acc_ood"] > t3["MViTv2-B"]["acc_ood"],
        "later_lower_pd_than_mvitv2": t3["LATER"]["pd"] < t3["MViTv2-B"]["pd"],
        "id_ood_matched_videos": float(cfg.id_test_videos == cfg.ood_test_videos),
        "later_h_beats_global_correction": t6["LATER"]["h"] > t6["Global Correction"]["h"],
        "later_id_beats_global_correction": t6["LATER"]["acc_id"] > t6["Global Correction"]["acc_id"],
        "large_id_ood_gap_mvitv2": t3["MViTv2-B"]["acc_id"] - t3["MViTv2-B"]["acc_ood"] > 30.0,
    }
