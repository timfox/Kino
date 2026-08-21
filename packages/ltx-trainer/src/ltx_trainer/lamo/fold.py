"""Fold LaMo motion-drift readouts into video latent shards (arXiv:2605.23878)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.lamo.config import LaMoConfig


def lamo_meta_block() -> dict[str, Any]:
    return {
        "lamo": {
            "arxiv_id": "2605.23878",
            "fold_role": "latent_motion_drift_proxy",
            "tau": LaMoConfig().tau,
        }
    }


def _latent_time_series(arr: np.ndarray) -> np.ndarray:
    """Return ``(T, C, H, W)`` from common LTX latent layouts."""
    if arr.ndim == 5:
        arr = arr[0]
    if arr.ndim != 4:
        raise ValueError(f"expected 4D or 5D latents, got shape {arr.shape}")
    # Heuristic: time is the smaller of the first two dims when one is small (frames).
    if arr.shape[0] <= arr.shape[1] and arr.shape[0] <= 64:
        return arr
    if arr.shape[1] <= 64:
        return np.transpose(arr, (1, 0, 2, 3))
    return arr


def motion_drift_norm_np(z: np.ndarray, *, tau: int = 2) -> float:
    """‖μ‖₂ for channel macro drift of τ-step latent deltas (Sec. 3.2 proxy)."""
    z = _latent_time_series(z)
    if z.shape[0] <= tau:
        return 0.0
    delta = z[tau:] - z[:-tau]
    mu = delta.mean(axis=(-2, -1))
    return float(np.linalg.norm(mu))


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    cfg = LaMoConfig()
    out = dict(data)
    out.update(lamo_meta_block())
    latents = data.get("latents")
    if latents is None:
        return out

    arr = np.asarray(latents.detach().cpu().float().numpy() if hasattr(latents, "detach") else latents)
    drift = motion_drift_norm_np(arr, tau=cfg.tau)
    t_len = int(_latent_time_series(arr).shape[0])
    out["lamo"].update(
        {
            "motion_drift_norm": round(drift, 5),
            "num_latent_frames": t_len,
            "strong_motion_frame": int(min(t_len - 1, max(0, t_len // 2))),
        }
    )
    return out
