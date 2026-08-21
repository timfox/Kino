"""Proximity attention point rendering stub (Sec. 3.2, Zhang et al. NeurIPS 2023)."""

from __future__ import annotations

from typing import Any

import numpy as np


def proximity_attention_weights(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    points: np.ndarray,
    *,
    top_k: int = 8,
    temperature: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select top-K points nearest the ray and softmax attention from inverse distance.
    Returns (indices, weights) each shape (top_k,).
    """
    ro = np.asarray(ray_origin, dtype=np.float64).reshape(3)
    rd = np.asarray(ray_dir, dtype=np.float64).reshape(3)
    rd = rd / (np.linalg.norm(rd) + 1e-12)
    pts = np.asarray(points, dtype=np.float64)
    # Perpendicular distance to ray
    v = pts - ro
    t_proj = (v @ rd).reshape(-1, 1)
    closest = ro + t_proj * rd
    perp = np.linalg.norm(pts - closest, axis=1)
    k = min(top_k, len(pts))
    idx = np.argsort(perp)[:k]
    scores = -perp[idx] / max(temperature, 1e-6)
    scores = scores - scores.max()
    w = np.exp(scores)
    w = w / (w.sum() + 1e-12)
    return idx, w


def render_patch(
    points: np.ndarray,
    features: np.ndarray,
    *,
    height: int = 32,
    width: int = 32,
    top_k: int = 8,
    seed: int = 0,
) -> np.ndarray:
    """Toy RGB patch: aggregate feature vectors along grid rays."""
    rng = np.random.default_rng(seed)
    pts = np.asarray(points, dtype=np.float64)
    feat = np.asarray(features, dtype=np.float64)
    rgb = np.zeros((height, width, 3), dtype=np.float64)
    cam = np.array([0.0, 0.0, 3.0])
    for y in range(height):
        for x in range(width):
            ndc_x = (x / max(width - 1, 1) - 0.5) * 2.0
            ndc_y = (0.5 - y / max(height - 1, 1)) * 2.0
            rd = np.array([ndc_x, ndc_y, -1.0])
            rd = rd / np.linalg.norm(rd)
            idx, w = proximity_attention_weights(cam, rd, pts, top_k=top_k)
            agg = (w.reshape(-1, 1) * feat[idx]).sum(axis=0)
            rgb[y, x] = 1.0 / (1.0 + np.exp(-agg[:3]))
    return rgb


def papr_render_demo(*, seed: int = 0, num_points: int = 256) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((num_points, 3))
    feat = rng.standard_normal((num_points, 16))
    rgb = render_patch(pts, feat, height=16, width=16, seed=seed)
    return {
        "patch_shape": list(rgb.shape),
        "patch_mean": float(rgb.mean()),
        "top_k": 8,
    }
