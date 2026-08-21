"""RGB point-cloud baseline readout: rasterize + VAE encode (Eq. 2)."""
from __future__ import annotations

import numpy as np

from ltx_trainer.mirage.memory import RGBPointCloudMemory


def rasterize_rgb(
    memory: RGBPointCloudMemory,
    k_pixel: np.ndarray,
    extrinsics: np.ndarray,
    *,
    rgb_hw: tuple[int, int],
) -> np.ndarray:
    """Z-buffer RGB rasterisation at pixel resolution."""
    H, W = rgb_hw
    image = np.zeros((3, H, W), dtype=np.float64)
    depth_buf = np.full((H, W), np.inf, dtype=np.float64)
    k = np.asarray(k_pixel, dtype=np.float64)
    for idx, p in enumerate(memory.points):
        ph = np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
        q = extrinsics @ ph
        if q[2] <= 1e-6:
            continue
        uv1 = k @ q[:3]
        uv1 = uv1 / uv1[2]
        u, v = int(np.floor(uv1[0])), int(np.floor(uv1[1]))
        if u < 0 or v < 0 or u >= W or v >= H:
            continue
        if q[2] < depth_buf[v, u]:
            depth_buf[v, u] = q[2]
            image[:, v, u] = memory.colors[idx]
    return image


def pseudo_vae_encode(rgb: np.ndarray, latent_hw: tuple[int, int]) -> np.ndarray:
    """Average-pool RGB to latent grid as VAE encode surrogate."""
    c, H, W = rgb.shape
    h, w = latent_hw
    out = np.zeros((c, h, w), dtype=np.float64)
    for yi in range(h):
        for xi in range(w):
            y0 = int(yi * H / h)
            y1 = max(y0 + 1, int((yi + 1) * H / h))
            x0 = int(xi * W / w)
            x1 = max(x0 + 1, int((xi + 1) * W / w))
            out[:, yi, xi] = rgb[:, y0:y1, x0:x1].mean(axis=(1, 2))
    return out


def rgb_memory_readout(
    memory: RGBPointCloudMemory,
    k_pixel: np.ndarray,
    extrinsics: np.ndarray,
    *,
    rgb_hw: tuple[int, int],
    latent_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Full pixel round trip: rasterise(M_rgb) → E(·) (Eq. 2)."""
    if not memory.points:
        c = 3
        h, w = latent_hw
        return np.zeros((c, h, w), dtype=np.float64), np.zeros((h, w), dtype=np.float64)
    rgb = rasterize_rgb(memory, k_pixel, extrinsics, rgb_hw=rgb_hw)
    z_hat = pseudo_vae_encode(rgb, latent_hw)
    mask = (np.abs(z_hat).sum(axis=0) > 1e-9).astype(np.float64)
    return z_hat, mask


def compare_readout_fidelity(
    source_latent: np.ndarray,
    latent_readout: np.ndarray,
    rgb_readout: np.ndarray,
    visibility: np.ndarray,
) -> dict[str, float]:
    """MSE on visible cells: latent cache vs RGB round trip."""
    vis = visibility > 0.5
    if not np.any(vis):
        return {"latent_mse": 0.0, "rgb_mse": 0.0, "rgb_penalty": 0.0}
    c = min(source_latent.shape[0], latent_readout.shape[0], rgb_readout.shape[0])
    src = source_latent[:c, vis]
    lat = latent_readout[:c, vis]
    rgb = rgb_readout[:c, vis]
    latent_mse = float(np.mean((src - lat) ** 2))
    rgb_mse = float(np.mean((src - rgb) ** 2))
    return {
        "latent_mse": latent_mse,
        "rgb_mse": rgb_mse,
        "rgb_penalty": rgb_mse - latent_mse,
    }
