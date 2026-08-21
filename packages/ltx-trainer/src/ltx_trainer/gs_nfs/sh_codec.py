"""SH decorrelation: RGB→YUV + per-channel KLT (Sec. 4)."""

from __future__ import annotations

import numpy as np

_RGB_TO_YUV = np.array(
    [
        [0.299, 0.587, 0.114],
        [-0.169, -0.331, 0.500],
        [0.500, -0.419, -0.081],
    ],
    dtype=np.float64,
)


def rgb_to_yuv(rgb: np.ndarray) -> np.ndarray:
    """Apply 3×3 RGB→YUV to last dimension."""
    flat = rgb.reshape(-1, 3)
    yuv = flat @ _RGB_TO_YUV.T
    return yuv.reshape(rgb.shape)


def klt_matrix(coeffs: np.ndarray) -> np.ndarray:
    """Empirical KLT from AC SH coefficient covariance."""
    x = coeffs.reshape(coeffs.shape[0], -1)
    cov = np.cov(x.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    return eigvecs[:, order]


def apply_klt(coeffs: np.ndarray, basis: np.ndarray) -> np.ndarray:
    flat = coeffs.reshape(coeffs.shape[0], -1)
    return (flat @ basis).reshape(coeffs.shape)


def decorrelate_sh(sh_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split DC vs AC; YUV+KLT on AC only."""
    dc = sh_rgb[..., :3]
    ac = sh_rgb[..., 3:]
    dc_yuv = rgb_to_yuv(dc)
    if ac.size == 0:
        return dc_yuv, np.eye(3)
    basis = klt_matrix(ac)
    ac_klt = apply_klt(ac, basis)
    out = np.concatenate([dc_yuv, ac_klt.reshape(ac.shape)], axis=-1)
    return out, basis


def compressed_size_proxy(
    attrs: np.ndarray,
    *,
    use_yuv_klt: bool,
    quant_step: float,
) -> float:
    """Entropy proxy: count non-zero quantized coeffs."""
    if use_yuv_klt and attrs.shape[-1] >= 3:
        sh = attrs.reshape(attrs.shape[0], -1, 3)
        attrs, _ = decorrelate_sh(sh)
        attrs = attrs.reshape(attrs.shape[0], -1)
    q = np.round(attrs / quant_step)
    nz = np.count_nonzero(q)
    return float(nz * np.log2(max(nz, 2)))
