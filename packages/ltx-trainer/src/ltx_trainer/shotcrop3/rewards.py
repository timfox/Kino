"""GRPO-S composite reward (Eq. 3–6)."""

from __future__ import annotations

import numpy as np

from ltx_trainer.shotcrop3.config import GRPOWeights
from ltx_trainer.shotcrop3.geometry import aspect_ratio, aspect_ratio_reward, iou


def aesthetic_reward_proxy(crop_features: np.ndarray) -> float:
    """Eq. 5 proxy: bounded aesthetic score from feature norms."""
    norm = float(np.linalg.norm(crop_features))
    return float(np.clip(norm / (norm + 0.35), 0.0, 1.0))


def shot_reward(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    crop_features: np.ndarray,
    weights: GRPOWeights | None = None,
) -> dict[str, float]:
    weights = weights or GRPOWeights()
    r_iou = iou(pred, gt)
    r_aes = aesthetic_reward_proxy(crop_features)
    r_ratio = aspect_ratio_reward(aspect_ratio(pred))
    total = weights.lambda_iou * r_iou + weights.lambda_aes * r_aes + weights.lambda_ratio * r_ratio
    return {
        "total": float(total),
        "iou": r_iou,
        "aesthetic": r_aes,
        "ratio": r_ratio,
    }


def trajectory_reward(
    preds: dict[str, np.ndarray],
    gts: dict[str, np.ndarray],
    features: dict[str, np.ndarray],
    *,
    weights: GRPOWeights | None = None,
) -> float:
    """Composite reward averaged over three TSC shots."""
    scores = [
        shot_reward(preds[k], gts[k], crop_features=features[k], weights=weights)["total"]
        for k in gts
    ]
    return float(np.mean(scores))
