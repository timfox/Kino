"""Runnable evaluation smoke for weatherproof."""

from __future__ import annotations

from typing import Any

from ltx_trainer.smoke_util import load_sibling


def evaluation_smoke() -> dict[str, Any]:
    try:
        import torch  # noqa: F401

        return _torch_smoke()
    except ImportError:
        return _numpy_smoke()


def _torch_smoke() -> dict[str, Any]:
    syn = load_sibling(__file__, "synthetic")
    met = load_sibling(__file__, "metrics")
    import torch
    _clean, _degraded, mask = syn.synthesize_pair(size=64)
    pred = mask.clone()
    tgt = mask.clone()
    iou = met.mean_iou(pred, tgt)
    return {"package": "weatherproof", "mean_iou": round(iou, 4)}


def _numpy_smoke() -> dict[str, Any]:
    import numpy as np
    pred = np.zeros((64, 64), dtype=np.int64)
    pred[16:48, 16:48] = 1
    tgt = pred.copy()
    inter = int((pred == tgt).sum())
    union = int(pred.size + tgt.size - inter)
    iou = inter / max(union, 1)
    return {"package": "weatherproof", "mean_iou": round(float(iou), 4)}
