"""KB high-pass residual extractor and covariance fingerprints."""

from __future__ import annotations

import numpy as np

KB_KERNEL = np.array(
    [[-1.0, 2.0, -1.0], [2.0, -4.0, 2.0], [-1.0, 2.0, -1.0]],
    dtype=np.float64,
) / 4.0


def kb_residual(image: np.ndarray) -> np.ndarray:
    """Apply KB filter E (paper Sec. 3.3)."""
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3:
        img = img.mean(axis=-1)
    from numpy.lib.stride_tricks import sliding_window_view

    patches = sliding_window_view(img, (3, 3))
    return np.tensordot(patches, KB_KERNEL, axes=((2, 3), (0, 1)))


def patch_residuals(
    image: np.ndarray,
    patch_shape: tuple[int, int] = (8, 16),
) -> np.ndarray:
    """Flatten 8×16 JPEG-grid-aligned patches into rows for covariance estimation."""
    r = kb_residual(image)
    h, w = patch_shape
    rows: list[np.ndarray] = []
    for y in range(0, r.shape[0] - h + 1, h):
        for x in range(0, r.shape[1] - w + 1, w):
            rows.append(r[y : y + h, x : x + w].reshape(-1))
    if not rows:
        return np.zeros((1, h * w), dtype=np.float64)
    return np.stack(rows, axis=0)


def residual_covariance(patches: np.ndarray) -> np.ndarray:
    """Sample covariance of residual patch vectors."""
    if patches.size == 0:
        return np.eye(1, dtype=np.float64) * 1e-6
    d = int(patches.shape[1])
    if patches.shape[0] < 2:
        return np.eye(d, dtype=np.float64) * 1e-6
    return np.cov(patches, rowvar=False) + np.eye(d, dtype=np.float64) * 1e-6


def frobenius_cov_distance(cov_a: np.ndarray, cov_b: np.ndarray) -> float:
    """|| Cov(A) − Cov(B) ||_F^2 (geometric alignment term)."""
    diff = cov_a - cov_b
    return float(np.sum(diff * diff))
