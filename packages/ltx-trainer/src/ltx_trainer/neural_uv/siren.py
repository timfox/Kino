"""Untrained SIREN for UV reparameterization (Eq. 2–3)."""

from __future__ import annotations

from typing import Any

import numpy as np


def siren_layer(x: np.ndarray, W: np.ndarray, b: np.ndarray, *, omega0: float = 15.0) -> np.ndarray:
    """h = sin(omega0 * (W @ x + b))."""
    return np.sin(omega0 * (x @ W.T + b))


def init_siren_weights(
    in_dim: int,
    *,
    layers: int = 5,
    width: int = 256,
    omega0: float = 15.0,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Xavier-style init for SIREN (first layer scaled by 1/omega0)."""
    rng = np.random.default_rng(seed)
    dims = [in_dim] + [width] * (layers - 1) + [2]
    params: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(len(dims) - 1):
        fan_in = dims[i]
        scale = np.sqrt(6.0 / fan_in)
        if i == 0:
            scale /= omega0
        W = rng.uniform(-scale, scale, size=(dims[i + 1], fan_in))
        b = np.zeros(dims[i + 1])
        params.append((W, b))
    return params


def siren_forward(features: np.ndarray, params: list[tuple[np.ndarray, np.ndarray]], *, omega0: float = 15.0) -> np.ndarray:
    """Map per-vertex features (N, D) → UV (N, 2)."""
    x = np.asarray(features, dtype=np.float64)
    for i, (W, b) in enumerate(params[:-1]):
        x = siren_layer(x, W, b, omega0=omega0)
    W_out, b_out = params[-1]
    return x @ W_out.T + b_out


def siren_demo(*, seed: int = 0, num_vertices: int = 64, feature_dim: int = 19) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    feats = rng.standard_normal((num_vertices, feature_dim))
    params = init_siren_weights(feature_dim, seed=seed)
    uv = siren_forward(feats, params)
    return {
        "uv_shape": list(uv.shape),
        "uv_mean": [float(uv[:, 0].mean()), float(uv[:, 1].mean())],
        "num_layers": len(params),
    }
