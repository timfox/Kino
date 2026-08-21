"""Synthetic async event VO simulation (Sec. IV-A)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ltx_trainer.event_vo.config import EventVOConfig
from ltx_trainer.event_vo.lie import rot_from_theta, transform_point
from ltx_trainer.event_vo.projection import project_landmark
from ltx_trainer.event_vo.rate import FeatureUpdate


@dataclass
class SimulationScene:
    landmarks: np.ndarray  # (M, 3)
    times: np.ndarray
    gt_positions: np.ndarray  # (T, 3)
    gt_thetas: list[np.ndarray]
    updates: list[FeatureUpdate]
    feature_to_lm: dict[int, int]


def generate_planar_scene(
    cfg: EventVOConfig,
    *,
    duration_s: float = 12.0,
    hz: float = 50.0,
    n_landmarks: int = 60,
    noise_px: float = 1.0,
    seed: int = 42,
) -> SimulationScene:
    rng = np.random.default_rng(seed)
    times = np.arange(0, duration_s, 1.0 / hz)
    # Landmarks on z=0 plane (homography-friendly)
    landmarks = np.zeros((n_landmarks, 3))
    landmarks[:, 0] = rng.uniform(-3, 3, n_landmarks)
    landmarks[:, 1] = rng.uniform(-2, 2, n_landmarks)
    landmarks[:, 2] = rng.uniform(4.0, 8.0, n_landmarks)

    gt_positions = []
    gt_thetas = []
    updates: list[FeatureUpdate] = []
    feature_to_lm = {i: i for i in range(n_landmarks)}

    omega = 0.35
    radius = 0.8
    for t in times:
        theta = np.array([0.05 * np.sin(omega * t), 0.08 * np.cos(omega * t), omega * t * 0.15])
        p = np.array([radius * np.cos(omega * t), radius * np.sin(omega * t), 0.0])
        gt_positions.append(p)
        gt_thetas.append(theta)
        R = rot_from_theta(theta)
        order = rng.permutation(n_landmarks)
        for fid in order:
            lm = landmarks[fid]
            pc = transform_point(R, p, lm)
            if pc[2] <= 0.2:
                continue
            uv = project_landmark(pc, cfg.fx, cfg.fy, cfg.cx, cfg.cy)
            uv += rng.normal(scale=noise_px, size=2)
            if not (5 < uv[0] < cfg.image_w - 5 and 5 < uv[1] < cfg.image_h - 5):
                continue
            updates.append(FeatureUpdate(fid, float(t), float(uv[0]), float(uv[1])))

    return SimulationScene(
        landmarks=landmarks,
        times=times,
        gt_positions=np.asarray(gt_positions),
        gt_thetas=gt_thetas,
        updates=updates,
        feature_to_lm=feature_to_lm,
    )
