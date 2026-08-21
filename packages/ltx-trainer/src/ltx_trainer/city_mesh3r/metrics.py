"""Surface reconstruction P/R/F1 and mesh quality metrics (Tables 1, 6, Sec. 9)."""

from __future__ import annotations

import torch
from torch import Tensor


def precision_recall_f1(
    tp: float,
    fp: float,
    fn: float,
) -> dict[str, float]:
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return {"precision": prec, "recall": rec, "f1": f1}


def triangle_aspect_ratio(vertices: Tensor, faces: Tensor) -> float:
    """Mean aspect ratio AR(f) = h / (2r) (Eq. 51–52)."""
    if faces.numel() == 0:
        return 0.0
    ars: list[float] = []
    for f in faces:
        v0, v1, v2 = vertices[f[0]], vertices[f[1]], vertices[f[2]]
        e = [
            (v1 - v0).norm(),
            (v2 - v1).norm(),
            (v0 - v2).norm(),
        ]
        h = max(e)
        s = sum(e) / 2
        area2 = s * (s - e[0]) * (s - e[1]) * (s - e[2])
        if area2 <= 1e-12:
            continue
        area = area2.sqrt()
        r = area / (e[0] + e[1] + e[2] + 1e-8) * 2  # inradius proxy
        ars.append(float(h / (2 * r + 1e-8)))
    return sum(ars) / max(len(ars), 1)


def mesh_quality_bundle(vertices: Tensor, faces: Tensor) -> dict[str, float]:
    """Toy mesh QA metrics (Sec. 9)."""
    return {
        "aspect_ratio": round(triangle_aspect_ratio(vertices, faces), 4),
        "num_vertices": float(vertices.shape[0]),
        "num_faces": float(faces.shape[0]),
    }
