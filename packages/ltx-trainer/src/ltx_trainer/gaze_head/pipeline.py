"""Framework card, paper tables, and smoke demos (Liu et al., arXiv:2605.25810)."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.gaze_head.baselines import constant_head_motion, mirror_gaze_inputs
from ltx_trainer.gaze_head.config import GazeHeadConfig
from ltx_trainer.gaze_head.metrics import (
    average_pairwise_distance,
    evaluate_sequence,
)
from ltx_trainer.gaze_head.model import GazeHeadCoordinationCVAE


def framework_card(cfg: GazeHeadConfig | None = None) -> dict[str, Any]:
    cfg = cfg or GazeHeadConfig()
    return {
        "name": "Gaze-Head Coordination cVAE",
        "paper": cfg.paper_arxiv,
        "doi": cfg.paper_doi,
        "venue": cfg.conference,
        "authors": "Liu, Wen, Sugano (UTokyo IIS)",
        "dataset": cfg.dataset,
        "train_videos": cfg.train_videos,
        "test_videos": cfg.test_videos,
        "seq_len": cfg.seq_len,
        "model_fps": cfg.model_fps,
        "latent_dim": cfg.latent_dim,
        "loss": "L = ||ĥ - h||² + λ L_KL (Eq. 1)",
        "temporal": "GRU encoder/decoder + autoregressive context c_{1:2}",
        "application": cfg.video_synthesis,
    }


def data_pipeline_stages() -> list[dict[str, str]]:
    """Fig. 1 — automatic in-the-wild extraction pipeline."""
    return [
        {"stage": "gaze_estimation", "tool": "appearance-based 3D gaze (Qin et al.)"},
        {"stage": "head_pose", "tool": "landmarks + SolvePnP + normalization"},
        {"stage": "filter_sunglasses", "tool": "Glasses Detector"},
        {"stage": "filter_multiface", "tool": "MediaPipe"},
        {"stage": "filter_stability", "tool": "PySceneDetect"},
    ]


def table_quantitative_comparison() -> dict[str, dict[str, float | None]]:
    """Table 1 — CelebV-Text test set (30 generations per gaze input)."""
    return {
        "Constant Head Motion": {
            "angular_error_avg_deg": 26.256,
            "angular_error_best_deg": 26.256,
            "correlation_pitch_best": None,
            "correlation_yaw_best": None,
            "ave_avg_deg2": 103.283,
            "smoothness_avg": None,
            "apd_deg2": None,
        },
        "Mirror Gaze Inputs": {
            "angular_error_avg_deg": 24.227,
            "angular_error_best_deg": 24.227,
            "correlation_pitch_best": 0.157,
            "correlation_yaw_best": 0.300,
            "ave_avg_deg2": 127.902,
            "smoothness_avg": 25.287,
            "apd_deg2": None,
        },
        "Ours": {
            "angular_error_avg_deg": 16.548,
            "angular_error_best_deg": 10.835,
            "correlation_pitch_best": 0.509,
            "correlation_yaw_best": 0.598,
            "ave_avg_deg2": 89.934,
            "smoothness_avg": 6.987,
            "apd_deg2": 264.463,
        },
        "w/o Temporal Modeling": {
            "angular_error_avg_deg": 18.218,
            "angular_error_best_deg": 15.600,
            "correlation_pitch_best": 0.338,
            "correlation_yaw_best": 0.433,
            "ave_avg_deg2": 139.245,
            "smoothness_avg": 66.267,
            "apd_deg2": 435.081,
        },
    }


def table_human_evaluation() -> dict[str, dict[str, float]]:
    """Table 2 — expert preference (-2 baseline .. +2 ours)."""
    return {
        "Constant Head Motion": {"mean_preference": 0.250, "std": 0.829, "p_value": 0.688},
        "Mirror Gaze Inputs": {"mean_preference": 1.500, "std": 0.707, "p_value": 0.016},
        "w/o Temporal Modeling": {"mean_preference": 1.625, "std": 0.484, "p_value": 0.008},
        "Driven by Real Motion": {"mean_preference": 1.000, "std": 0.707, "p_value": 0.031},
    }


def training_step_demo(cfg: GazeHeadConfig | None = None) -> dict[str, float]:
    """Smoke: cVAE forward + loss on synthetic gaze/head sequences."""
    cfg = cfg or GazeHeadConfig()
    torch.manual_seed(11)
    b, t = 2, cfg.seq_len
    gaze = torch.randn(b, t, 2) * 0.2
    head = gaze * 0.5 + torch.randn(b, t, 2) * 0.05
    ctx = torch.zeros(b, cfg.context_frames, 2)
    model = GazeHeadCoordinationCVAE(cfg)
    loss, parts = model.training_loss(gaze, head, ctx)
    pred = model.sample_head(gaze, ctx)
    return {
        "loss": float(loss.detach()),
        "recon": parts["recon"],
        "kl": parts["kl"],
        "pred_shape_t": float(pred.shape[1]),
        "latent_dim": float(cfg.latent_dim),
    }


def evaluation_demo(cfg: GazeHeadConfig | None = None) -> dict[str, Any]:
    """Smoke: baselines vs correlated head on synthetic sequence; paper table checks."""
    cfg = cfg or GazeHeadConfig()
    step = training_step_demo(cfg)
    torch.manual_seed(3)
    t = cfg.seq_len
    gaze = torch.linspace(-0.3, 0.4, t).unsqueeze(1).expand(t, 2).clone()
    gaze[:, 1] = torch.linspace(-0.2, 0.25, t)
    head = gaze * 0.55 + torch.randn(t, 2) * 0.03

    const = constant_head_motion(gaze, head[0])
    mirror = mirror_gaze_inputs(gaze)

    m_const = evaluate_sequence(const, head)
    m_mirror = evaluate_sequence(mirror, head)
    m_corr = evaluate_sequence(head, head)

    gens = head.unsqueeze(0) + torch.randn(5, t, 2) * 0.02
    apd = average_pairwise_distance(gens)

    tab = table_quantitative_comparison()
    ours = tab["Ours"]
    mirror_tab = tab["Mirror Gaze Inputs"]

    return {
        **step,
        "baseline_const_angular_deg": m_const["angular_error_deg"],
        "baseline_mirror_angular_deg": m_mirror["angular_error_deg"],
        "correlated_head_angular_deg": m_corr["angular_error_deg"],
        "mirror_worse_than_correlated": m_mirror["angular_error_deg"] > m_corr["angular_error_deg"],
        "apd_smoke": apd,
        "paper_ours_best_angular_deg": ours["angular_error_best_deg"],
        "paper_mirror_best_angular_deg": mirror_tab["angular_error_best_deg"],
        "ours_beats_mirror_on_angular": float(ours["angular_error_best_deg"])
        < float(mirror_tab["angular_error_best_deg"]),
        "human_pref_vs_mirror": table_human_evaluation()["Mirror Gaze Inputs"]["mean_preference"],
        "data_pipeline_stages": len(data_pipeline_stages()),
    }
