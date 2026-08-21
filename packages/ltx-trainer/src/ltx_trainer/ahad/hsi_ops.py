"""HSI tensor utilities: shifting, ASF feature projection, gradients."""

from __future__ import annotations

import numpy as np

Shift = tuple[int, int]


def shift_tensor(y: np.ndarray, delta: Shift, *, mode: str = "edge") -> np.ndarray:
    """T_Δ operator (Eq. 13) with edge padding."""
    if y.ndim == 2:
        dh, dw = delta
        out = np.roll(y, shift=(dh, dw), axis=(0, 1))
        if dh > 0:
            out[:dh, :] = out[dh : dh + 1, :]
        elif dh < 0:
            out[dh:, :] = out[dh - 1 : dh, :]
        if dw > 0:
            out[:, :dw] = out[:, dw : dw + 1]
        elif dw < 0:
            out[:, dw:] = out[:, dw - 1 : dw]
        return out
    dh, dw = delta
    out = np.roll(y, shift=(dh, dw), axis=(0, 1))
    if dh > 0:
        out[:dh, :, :] = out[dh : dh + 1, :, :]
    elif dh < 0:
        out[dh:, :, :] = out[dh - 1 : dh, :, :]
    if dw > 0:
        out[:, :dw, :] = out[:, dw : dw + 1, :]
    elif dw < 0:
        out[:, dw:, :] = out[:, dw - 1 : dw, :]
    return np.clip(out, 0.0, 1.0)


def shift_set(radius: int) -> list[Shift]:
    s = range(-radius, radius + 1)
    return [(h, w) for h in s for w in s]


def mean_spectrum(y: np.ndarray) -> np.ndarray:
    return y.mean(axis=(0, 1), keepdims=False)


def mean_shift(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = y.mean(axis=(0, 1), keepdims=True)
    return y - mu, mu


def affine_subspace_basis(y: np.ndarray, n: int) -> np.ndarray:
    """Top-N left singular vectors of mode-3 unfolding (ASF)."""
    _, _, c = y.shape
    mat = y.reshape(-1, c)
    _, _, vt = np.linalg.svd(mat, full_matrices=False)
    return vt[:n].T


def project_features(y: np.ndarray, u: np.ndarray) -> np.ndarray:
    """eZ = eY_s ×_3 U."""
    h, w, c = y.shape
    flat = y.reshape(-1, c) @ u
    return flat.reshape(h, w, u.shape[1])


def grad_h(x: np.ndarray) -> np.ndarray:
    return np.diff(x, axis=0, prepend=x[:1, :, :])


def grad_w(x: np.ndarray) -> np.ndarray:
    return np.diff(x, axis=1, prepend=x[:, :1, :])


def grad_c(x: np.ndarray) -> np.ndarray:
    return np.diff(x, axis=2, prepend=x[:, :, :1])


def frobenius_norm(x: np.ndarray) -> float:
    return float(np.sqrt(np.sum(x**2)))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))
