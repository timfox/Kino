"""Source safety directions, transfer, and inference steering (Sec. 3)."""

from __future__ import annotations

import numpy as np

from ltx_trainer.cross_model_safety.alignment import apply_map, median_scale_ratio


def mean_pool_tokens(h: np.ndarray) -> np.ndarray:
    """μ(·) mean pool over token dimension."""
    if h.ndim == 1:
        return h
    if h.ndim == 2:
        return h.mean(axis=0)
    return h.mean(axis=tuple(range(h.ndim - 1)))


def source_safety_direction(h_safe: np.ndarray, h_unsafe: np.ndarray) -> np.ndarray:
    """v_s = mean_i μ(h(x+)) - μ(h(x-)) (Eq. 1–2)."""
    deltas = mean_pool_tokens(h_safe) - mean_pool_tokens(h_unsafe)
    if deltas.ndim == 2:
        return deltas.mean(axis=0)
    return deltas


def category_directions(
    safe_by_cat: dict[str, np.ndarray],
    unsafe_by_cat: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Multi-vector v_{s,c} (Eq. 11)."""
    out: dict[str, np.ndarray] = {}
    for cat in safe_by_cat:
        if cat not in unsafe_by_cat:
            continue
        out[cat] = source_safety_direction(safe_by_cat[cat], unsafe_by_cat[cat])
    return out


def calibrate_transferred_direction(
    v_source: np.ndarray,
    v_raw_target: np.ndarray,
    beta: float,
) -> np.ndarray:
    """v_t = β||v_s|| * ṽ/||ṽ|| (Eq. 9)."""
    norm_s = float(np.linalg.norm(v_source))
    norm_raw = float(np.linalg.norm(v_raw_target))
    if norm_raw < 1e-12:
        return np.zeros_like(v_raw_target)
    return beta * norm_s * (v_raw_target / norm_raw)


def transfer_direction(
    v_source: np.ndarray,
    hs_anchors: np.ndarray,
    ht_anchors: np.ndarray,
    method: str,
    map_params: object,
) -> np.ndarray:
    """Apply T_{s→t} + magnitude calibration (Eq. 7–9)."""
    beta = median_scale_ratio(hs_anchors, ht_anchors)
    raw = apply_map(v_source, method, map_params)
    return calibrate_transferred_direction(v_source, raw, beta)


def steer_hidden(
    h: np.ndarray,
    v_target: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    """ĥ = h + α v_t (Eq. 10)."""
    return h + alpha * v_target


def steer_multi_category(
    h: np.ndarray,
    directions: dict[str, np.ndarray],
    active: set[str],
    *,
    alpha: float,
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """Eq. 13 multi-vector average."""
    cats = [c for c in active if c in directions]
    if not cats:
        return h
    if weights is None:
        w = 1.0 / len(cats)
        v = sum(w * directions[c] for c in cats)
    else:
        total = sum(weights.get(c, 0.0) for c in cats) or 1.0
        v = sum((weights.get(c, 0.0) / total) * directions[c] for c in cats)
    return steer_hidden(h, v, alpha=alpha)
