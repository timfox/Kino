"""Bounding-box geometry for TSC (IoU, BDE, aspect ratio)."""

from __future__ import annotations

import math

import numpy as np


def xywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    x, y, w, h = box
    return np.array([x, y, x + w, y + h], dtype=float)


def xyxy_to_xywh(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([x1, y1, x2 - x1, y2 - y1], dtype=float)


def clip_xyxy(box: np.ndarray, *, width: float, height: float) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array(
        [
            np.clip(x1, 0.0, width),
            np.clip(y1, 0.0, height),
            np.clip(x2, 0.0, width),
            np.clip(y2, 0.0, height),
        ],
        dtype=float,
    )


def box_area(box: np.ndarray) -> float:
    x1, y1, x2, y2 = box
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Eq. 4 intersection-over-union."""
    px1, py1, px2, py2 = pred
    gx1, gy1, gx2, gy2 = gt
    ix1, iy1 = max(px1, gx1), max(py1, gy1)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area(pred) + box_area(gt) - inter
    return float(inter / (union + 1e-8))


def boundary_displacement_error(pred: np.ndarray, gt: np.ndarray) -> float:
    """Average Euclidean distance between box boundaries (BDE proxy)."""
    diffs = np.abs(pred - gt)
    return float(np.mean(diffs))


def aspect_ratio(box: np.ndarray) -> float:
    x1, y1, x2, y2 = box
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    return float(w / h)


def aspect_ratio_reward(r: float) -> float:
    """Eq. 6 cinematic 4:3 / 3:4 reward in [0, 1]."""
    if r >= 1.0:
        return max(0.0, 1.0 - abs(math.log(r) - math.log(4.0 / 3.0)))
    return max(0.0, 1.0 - abs(math.log(r) - math.log(3.0 / 4.0)))


def jitter_box(box: np.ndarray, *, scale: float, rng: np.random.Generator) -> np.ndarray:
    """Perturb a box for synthetic model proposals."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = x2 - x1, y2 - y1
    dw = rng.normal(scale=scale * w)
    dh = rng.normal(scale=scale * h)
    dcx = rng.normal(scale=scale * w)
    dcy = rng.normal(scale=scale * h)
    nw, nh = max(w + dw, 8.0), max(h + dh, 8.0)
    out = np.array([cx + dcx - nw / 2, cy + dcy - nh / 2, cx + dcx + nw / 2, cy + dcy + nh / 2])
    return out
