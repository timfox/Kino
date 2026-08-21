"""Fold S3PO WSS-L1 ERP super-resolution proxy into video latent shards."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.s3po.config import NUM_INPUT_FRAMES, PAPER_ARXIV, SCALE_FACTOR, S3POConfig


def s3po_meta_block() -> dict[str, Any]:
    return {
        "s3po": {
            "arxiv_id": PAPER_ARXIV,
            "fold_role": "erp_vsr_wss_proxy",
            "reference_metric": "WSS-L1",
            "scale_factor": SCALE_FACTOR,
        }
    }


def _latent_time_series(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 5:
        arr = arr[0]
    if arr.ndim != 4:
        raise ValueError(f"expected 4D or 5D latents, got shape {arr.shape}")
    if arr.shape[0] <= arr.shape[1] and arr.shape[0] <= 64:
        return arr
    if arr.shape[1] <= 64:
        return np.transpose(arr, (1, 0, 2, 3))
    return arr


def _erp_cos_weights(h: int) -> np.ndarray:
    i = np.arange(h, dtype=np.float32)
    return np.cos((i + 0.5 - h / 2.0) * np.pi / h).clip(1e-6, None)


def wss_quality_proxy_from_latents(z: np.ndarray) -> tuple[float, float]:
    """Return ``(wss_quality_proxy 1–5, wss_l1_residual_proxy)`` from temporal ERP latents."""
    z = _latent_time_series(z)
    t, _, h, w = z.shape
    if t < 2:
        return 3.0, 0.5

    diffs = []
    for t_idx in range(t - 1):
        a = z[t_idx].mean(axis=0)
        b = z[t_idx + 1].mean(axis=0)
        row_diff = np.abs(a - b).mean(axis=-1)
        psi = _erp_cos_weights(h)
        w = psi / psi.sum()
        diffs.append(float((row_diff * w).sum()))

    residual = float(np.mean(diffs))
    # Lower frame-to-frame WSS residual → higher VSR quality proxy
    quality = float(np.clip(5.0 - residual * 8.0, 1.0, 5.0))
    return quality, residual


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    cfg = S3POConfig()
    out = dict(data)
    out.update(s3po_meta_block())
    latents = data.get("latents")
    if latents is None:
        return out

    arr = np.asarray(latents.detach().cpu().float().numpy() if hasattr(latents, "detach") else latents)
    quality, residual = wss_quality_proxy_from_latents(arr)
    h = int(arr.shape[-2]) if arr.ndim >= 2 else cfg.lr_height
    w = int(arr.shape[-1]) if arr.ndim >= 1 else cfg.lr_width
    erp_like = h > 0 and w > 0 and abs((h / w) - 0.5) < 0.35

    out["s3po"].update(
        {
            "wss_quality_proxy": round(quality, 3),
            "mos_proxy": round(quality, 3),
            "wss_l1_residual_proxy": round(residual, 5),
            "erp_aspect_like": erp_like,
            "num_input_frames_ref": NUM_INPUT_FRAMES,
        }
    )
    return out
