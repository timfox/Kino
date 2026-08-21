"""Loop closure gating and covariance adjustment."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ltx_trainer.ground_texture_slam.jih import jih_symmetry_score, joint_intensity_histogram, scale_covariance_jih
from ltx_trainer.ground_texture_slam.kld import kld_rgb, scale_covariance_kld
from ltx_trainer.ground_texture_slam.overlap import visual_overlap
from ltx_trainer.ground_texture_slam.schema import LoopClosureMethod, Pose2D


@dataclass
class LoopCandidate:
    pose_a: Pose2D
    pose_b: Pose2D
    image_a: Tensor
    image_b: Tensor
    is_true_loop: bool = False
    accepted: bool = False
    weight: float = 1.0


def default_covariance() -> Tensor:
    return torch.eye(3) * 0.01


def evaluate_loop_candidate(
    cand: LoopCandidate,
    *,
    method: LoopClosureMethod,
    baseline: Tensor | None = None,
    min_overlap: float = 0.01,
) -> tuple[bool, Tensor]:
    """Return (accept, scaled_covariance)."""
    cov = default_covariance()
    if method == LoopClosureMethod.ODOMETRY:
        return False, cov
    if method == LoopClosureMethod.VISUAL_OVERLAP:
        ok = visual_overlap(cand.pose_a, cand.pose_b, min_area=min_overlap)
        return ok, cov
    if method in (LoopClosureMethod.KLD, LoopClosureMethod.KLD_GRAY):
        if baseline is None:
            return True, cov
        gray = method == LoopClosureMethod.KLD_GRAY
        score = kld_rgb(cand.image_b, baseline, grayscale=gray)
        # reject high dissimilarity loops (proxy: very high KLD)
        if score > 2.5:
            return False, scale_covariance_kld(cov, score)
        return True, scale_covariance_kld(cov, score)
    if method == LoopClosureMethod.JIH:
        h = joint_intensity_histogram(cand.image_a, cand.image_b)
        jih = jih_symmetry_score(h)
        if jih < 0.15:
            return False, scale_covariance_jih(cov, jih)
        return True, scale_covariance_jih(cov, jih)
    # original single/many: accept all candidates (prone to false positives)
    return True, cov
