"""2D DCT / IDCT (type-II / type-III) per SKILD paper Appendix C."""

from __future__ import annotations

import numpy as np


def dct2(x: np.ndarray) -> np.ndarray:
    """Forward 2D DCT on ``(H, W)`` float array."""
    try:
        from scipy.fft import dctn  # noqa: PLC0415

        return dctn(x, type=2, norm="ortho")
    except ImportError:
        return _dct2_separable(x)


def idct2(x: np.ndarray) -> np.ndarray:
    """Inverse 2D DCT."""
    try:
        from scipy.fft import idctn  # noqa: PLC0415

        return idctn(x, type=2, norm="ortho")
    except ImportError:
        return _idct2_separable(x)


def _dct1(a: np.ndarray) -> np.ndarray:
    n = a.shape[-1]
    out = np.empty_like(a)
    for u in range(n):
        cu = np.sqrt(1.0 / n) if u == 0 else np.sqrt(2.0 / n)
        xs = np.arange(n, dtype=np.float64)
        out[..., u] = cu * np.sum(a * np.cos(np.pi * (xs + 0.5) * u / n), axis=-1)
    return out


def _idct1(a: np.ndarray) -> np.ndarray:
    n = a.shape[-1]
    out = np.empty_like(a)
    xs = np.arange(n, dtype=np.float64)
    for i in range(n):
        acc = a[..., 0] / np.sqrt(n)
        for u in range(1, n):
            acc = acc + np.sqrt(2.0 / n) * a[..., u] * np.cos(np.pi * (xs + 0.5) * u / n)
        out[..., i] = acc
    return out


def _dct2_separable(x: np.ndarray) -> np.ndarray:
    t = _dct1(x)
    return np.transpose(_dct1(np.transpose(t, (1, 0))), (1, 0))


def _idct2_separable(x: np.ndarray) -> np.ndarray:
    t = _idct1(x)
    return np.transpose(_idct1(np.transpose(t, (1, 0))), (1, 0))


def frequency_radius_grid(h: int, w: int) -> np.ndarray:
    """Radial frequency magnitude ``‖k‖`` for each DCT mode, shape ``(H, W)``."""
    u = np.arange(h, dtype=np.float64) * np.pi
    v = np.arange(w, dtype=np.float64) * np.pi
    uu, vv = np.meshgrid(u, v, indexing="ij")
    return np.sqrt(uu * uu + vv * vv)
