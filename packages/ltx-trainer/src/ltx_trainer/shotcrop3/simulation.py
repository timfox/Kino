"""Synthetic TSC scenes, training stages, and PGS demo."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.shotcrop3.config import ShotCrop3Config, ShotCrop3Params
from ltx_trainer.shotcrop3.geometry import clip_xyxy, jitter_box
from ltx_trainer.shotcrop3.metrics import evaluate_triple, overall_score
from ltx_trainer.shotcrop3.pgs import select_pseudo_label
from ltx_trainer.shotcrop3.rewards import trajectory_reward


def synthetic_scene(
    *,
    width: float = 1024.0,
    height: float = 768.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Human-centric scene with expert triple-shot boxes."""
    rng = np.random.default_rng(seed)
    subject_cx = width * (0.45 + 0.05 * rng.standard_normal())
    subject_cy = height * (0.52 + 0.04 * rng.standard_normal())
    body_w, body_h = width * 0.38, height * 0.62
    medium = clip_xyxy(
        np.array(
            [
                subject_cx - body_w / 2,
                subject_cy - body_h / 2,
                subject_cx + body_w / 2,
                subject_cy + body_h / 2,
            ]
        ),
        width=width,
        height=height,
    )
    close_w, close_h = body_w * 0.45, body_h * 0.38
    close = clip_xyxy(
        np.array(
            [
                subject_cx - close_w / 2,
                subject_cy - close_h * 0.15,
                subject_cx + close_w / 2,
                subject_cy - close_h * 0.15 + close_h,
            ]
        ),
        width=width,
        height=height,
    )
    env_cx = width * (0.22 + 0.03 * rng.standard_normal())
    env_cy = height * (0.28 + 0.03 * rng.standard_normal())
    env_w, env_h = width * 0.36, height * 0.34
    establishing = clip_xyxy(
        np.array([env_cx - env_w / 2, env_cy - env_h / 2, env_cx + env_w / 2, env_cy + env_h / 2]),
        width=width,
        height=height,
    )
    gts = {"medium": medium, "close_up": close, "establishing": establishing}
    features = {k: rng.normal(size=32) + 0.5 for k in gts}
    return {"width": width, "height": height, "gts": gts, "features": features}


def predict_for_stage(
    gts: dict[str, np.ndarray],
    stage: str,
    *,
    width: float,
    height: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """Noisy proposals improving across CoT-SFT → Semi-SFT → GRPO."""
    scales = {
        "base": 0.28,
        "cot_sft": 0.16,
        "semi_sft": 0.11,
        "grpo": 0.05,
    }
    scale = scales.get(stage, 0.2)
    rng = np.random.default_rng(seed)
    return {
        k: clip_xyxy(jitter_box(box, scale=scale, rng=rng), width=width, height=height)
        for k, box in gts.items()
    }


def stage_metrics(
    scene: dict[str, Any],
    stage: str,
    *,
    seed: int = 0,
) -> dict[str, float]:
    preds = predict_for_stage(
        scene["gts"],
        stage,
        width=scene["width"],
        height=scene["height"],
        seed=seed,
    )
    geom = evaluate_triple(preds, scene["gts"])
    reward = trajectory_reward(preds, scene["gts"], scene["features"])
    scores = overall_score(
        iou_mean=geom["iou"],
        aesthetic=0.72 + 0.08 * geom["iou"],
        storytelling=0.55 + 0.10 * geom["iou"],
    )
    return {**geom, **scores, "grpo_reward": reward}


def benchmark_synthetic(
    *,
    n_scenes: int = 48,
    seed: int = 7,
    params: ShotCrop3Params | None = None,
) -> dict[str, dict[str, float]]:
    params = params or ShotCrop3Params()
    stages = ("base", "cot_sft", "semi_sft", "grpo")
    acc: dict[str, list[dict[str, float]]] = {s: [] for s in stages}
    for i in range(n_scenes):
        scene = synthetic_scene(seed=seed + i)
        for st in stages:
            acc[st].append(stage_metrics(scene, st, seed=seed + 100 + i))
    return {
        st: {k: float(np.mean([row[k] for row in rows])) for k in rows[0]}
        for st, rows in acc.items()
    }


def pgs_demo(scene: dict[str, Any], *, seed: int = 0, cfg: ShotCrop3Config | None = None) -> dict[str, Any]:
    cfg = cfg or ShotCrop3Config()
    rng = np.random.default_rng(seed)
    width, height = scene["width"], scene["height"]
    qualities = {"base": 0.35, "sft": 0.55, "cot_sft": 0.72}
    counts = {"pseudo": 0, "hard": 0, "reject": 0}
    selected_models: list[str] = []
    for shot in cfg.shot_types:
        proposals = {
            "base": predict_for_stage(scene["gts"], "base", width=width, height=height, seed=seed + 1)[shot],
            "sft": predict_for_stage(scene["gts"], "cot_sft", width=width, height=height, seed=seed + 2)[shot],
            "cot_sft": predict_for_stage(scene["gts"], "semi_sft", width=width, height=height, seed=seed + 3)[shot],
        }
        name, status = select_pseudo_label(
            proposals,
            scene["gts"],
            shot_type=shot,
            model_qualities=qualities,
            thresholds=cfg.params.pgs,
            rng=rng,
        )
        counts[status] += 1
        if name:
            selected_models.append(name)
    return {"counts": counts, "selected_models": selected_models}


def ablation_from_synthetic(seed: int = 7) -> list[dict[str, float]]:
    """Map synthetic stage metrics to Table 2 ordering."""
    syn = benchmark_synthetic(seed=seed)
    return [
        {"stage": "base", **{k: syn["base"][k] for k in ("iou", "bde", "overall")}},
        {"stage": "cot_sft", **{k: syn["cot_sft"][k] for k in ("iou", "bde", "overall")}},
        {"stage": "semi_sft", **{k: syn["semi_sft"][k] for k in ("iou", "bde", "overall")}},
        {"stage": "grpo", **{k: syn["grpo"][k] for k in ("iou", "bde", "overall", "storytelling")}},
    ]
