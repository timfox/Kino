"""Fold PhyWorld physics-faithfulness proxy into video latent shards (arXiv:2605.19242)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.phyworld.judge import overall_physics_score


def phyworld_meta_block() -> dict[str, Any]:
    return {
        "phyworld": {
            "arxiv_id": "2605.19242",
            "fold_role": "temporal_physics_proxy",
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


def physics_proxy_from_latents(z: np.ndarray) -> tuple[float, float]:
    """Return ``(physics_proxy_1_5, jerk_std)`` from temporal smoothness of latent trajectories."""
    z = _latent_time_series(z)
    if z.shape[0] < 3:
        return 3.0, 0.0
    frame_energy = z.reshape(z.shape[0], -1).std(axis=1)
    vel = np.diff(frame_energy)
    jerk = np.diff(vel)
    jerk_std = float(jerk.std()) if jerk.size else 0.0
    smoothness = 1.0 / (1.0 + jerk_std)
    sa = float(np.clip(0.6 + 0.35 * smoothness, 0.0, 1.0))
    ptv = float(np.clip(1.0 - jerk_std * 2.0, 0.0, 1.0))
    persist = float(np.clip(1.0 - abs(frame_energy.std() - frame_energy.mean()) / (frame_energy.mean() + 1e-6), 0.0, 1.0))
    solid = float(np.clip(smoothness, 0.0, 1.0))
    fluid = ptv
    optical = sa
    overall = overall_physics_score(sa * 100, ptv * 100, persist * 100, solid * 100, fluid * 100, optical * 100)
    proxy_1_5 = float(np.clip(1.0 + (overall / 100.0) * 4.0, 1.0, 5.0))
    return proxy_1_5, jerk_std


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    out.update(phyworld_meta_block())
    latents = data.get("latents")
    if latents is None:
        return out

    arr = np.asarray(latents.detach().cpu().float().numpy() if hasattr(latents, "detach") else latents)
    proxy, jerk = physics_proxy_from_latents(arr)
    out["phyworld"].update(
        {
            "physics_proxy": round(proxy, 3),
            "temporal_jerk_std": round(jerk, 5),
        }
    )
    return out
