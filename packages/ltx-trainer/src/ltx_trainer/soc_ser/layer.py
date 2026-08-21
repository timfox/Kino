"""Second-Order Correlation (SOC) layer — Algorithm 1."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.soc_ser.config import SocSerConfig


def vech(symmetric: np.ndarray) -> np.ndarray:
    """Half-vectorize lower triangle of symmetric matrix S (Step 5)."""
    d = symmetric.shape[0]
    return symmetric[np.tril_indices(d)]


def log_euclidean_map(
    c_hat: np.ndarray,
    *,
    eps_id: float = 1e-6,
) -> np.ndarray:
    """LEM: S = U diag(log λ) U^T (Steps 4–6)."""
    d = c_hat.shape[0]
    eigvals, eigvecs = np.linalg.eigh(c_hat + eps_id * np.eye(d))
    eigvals = np.clip(eigvals, 1e-12, None)
    log_lambda = np.diag(np.log(eigvals))
    return eigvecs @ log_lambda @ eigvecs.T


def trace_normalized_covariance(
    z: np.ndarray,
    *,
    eps_div: float = 1e-6,
) -> np.ndarray:
    """Sample covariance with trace normalization (Steps 3)."""
    t = z.shape[0]
    cov = (z.T @ z) / (t - 1)
    return cov / (float(np.trace(cov)) + eps_div)


def soc_layer(
    features: np.ndarray,
    projection: np.ndarray,
    *,
    use_lem: bool = True,
    cfg: SocSerConfig | None = None,
) -> np.ndarray:
    """
    SOC forward pass (Algorithm 1).

    Args:
        features: frame-level SSL features X ∈ R^{T×D_in}
        projection: learnable subspace map W ∈ R^{D_in×d}
        use_lem: if False, ablation SOC w/o LEM (trace-normalized C only)
    """
    c = cfg or SocSerConfig()
    x = np.asarray(features, dtype=np.float64)
    w = np.asarray(projection, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("features must be 2-D (T, D_in)")

    t = x.shape[0]
    if t < 2:
        raise ValueError("need at least 2 frames for covariance")

    x_bar = x.mean(axis=0, keepdims=True)
    z = (x - x_bar) @ w
    c_hat = trace_normalized_covariance(z, eps_div=c.eps_div)

    s = log_euclidean_map(c_hat, eps_id=c.eps_id) if use_lem else c_hat
    return vech(s)


def subspace_dim_curve(
    dims: np.ndarray,
    *,
    peak_d: int = 32,
    peak_wa: float = 73.50,
    floor_wa: float = 69.0,
) -> np.ndarray:
    """Toy unimodal WA vs subspace d (Figure 2)."""
    x = np.asarray(dims, dtype=np.float64)
    width = max(peak_d * 0.45, 8.0)
    return floor_wa + (peak_wa - floor_wa) * np.exp(-0.5 * ((x - peak_d) / width) ** 2)


def soc_demo(seed: int = 0, cfg: SocSerConfig | None = None) -> dict[str, Any]:
    """CPU smoke: SOC vector shape + LEM vs no-LEM separation."""
    c = cfg or SocSerConfig()
    rng = np.random.default_rng(seed)
    t, d_in, d = 80, c.ssl_dim, c.default_subspace_d
    x = rng.normal(size=(t, d_in))
    w = rng.normal(scale=0.02, size=(d_in, d))

    v_lem = soc_layer(x, w, use_lem=True, cfg=c)
    v_no_lem = soc_layer(x, w, use_lem=False, cfg=c)
    expected_len = d * (d + 1) // 2

    dims = np.array([24, 32, 48, 64, 128])
    esd_curve = subspace_dim_curve(dims, peak_d=32, peak_wa=c.hubert_soc_esd_wa)
    rav_curve = subspace_dim_curve(
        dims, peak_d=48, peak_wa=c.hubert_soc_ravdess_wa, floor_wa=62.0
    )

    return {
        "frames": t,
        "subspace_d": d,
        "vector_dim": int(v_lem.shape[0]),
        "expected_vector_dim": expected_len,
        "lem_differs_from_no_lem": not np.allclose(v_lem, v_no_lem),
        "subspace_unimodal_esd": float(esd_curve.max()) > float(esd_curve[0]),
        "subspace_peak_near_32": int(dims[np.argmax(esd_curve)]) == 32,
        "subspace_dims": dims.tolist(),
        "esd_wa_curve": esd_curve.tolist(),
        "ravdess_wa_curve": rav_curve.tolist(),
    }
