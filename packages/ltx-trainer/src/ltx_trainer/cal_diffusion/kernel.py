"""Diffusion kernel proxy for Eq. (1) — steady-state cubic-sink blur."""

from __future__ import annotations

import numpy as np


def diffusion_length_m(*, d_m2_s: float, t_s: float) -> float:
    """Characteristic diffusion length sqrt(4 D t)."""
    return float(np.sqrt(max(4.0 * d_m2_s * t_s, 0.0)))


def gaussian_kernel_3d(
    shape: tuple[int, int, int],
    *,
    d_m2_s: float,
    t_s: float,
    voxel_m: float,
    truncate_sigma: float = 3.0,
) -> np.ndarray:
    """Separable Gaussian approximation to Eq. (1) for CPU stub."""
    sigma_m = diffusion_length_m(d_m2_s=d_m2_s, t_s=t_s)
    if sigma_m <= 0.0:
        k = np.zeros(shape, dtype=np.float64)
        k[shape[0] // 2, shape[1] // 2, shape[2] // 2] = 1.0
        return k

    sigma_vox = max(sigma_m / voxel_m, 0.5)
    radius = int(np.ceil(truncate_sigma * sigma_vox))
    coords = np.arange(-radius, radius + 1, dtype=np.float64)
    g1 = np.exp(-0.5 * (coords / sigma_vox) ** 2)
    g1 /= g1.sum()
    k = np.outer(g1, g1).reshape(-1, 1) * g1.reshape(1, 1, -1)
    k = k.reshape(len(g1), len(g1), len(g1))
    k /= k.sum()
    return k.astype(np.float64)


def convolve_volume(volume: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """FFT convolution with zero padding (same shape)."""
    from numpy.fft import fftn, ifftn

    pad = tuple(k - 1 for k in kernel.shape)
    full = np.pad(volume, [(p // 2, p - p // 2) for p in pad], mode="constant")
    k_pad = np.zeros_like(full)
    off = [(full.shape[i] - kernel.shape[i]) // 2 for i in range(3)]
    k_pad[off[0] : off[0] + kernel.shape[0], off[1] : off[1] + kernel.shape[1], off[2] : off[2] + kernel.shape[2]] = kernel
    out = np.real(ifftn(fftn(full) * fftn(k_pad)))
    sl = tuple(slice(p // 2, p // 2 + volume.shape[i]) for i, p in enumerate(pad))
    return out[sl].astype(np.float64)


def wiener_deconvolve(
    signal: np.ndarray,
    kernel: np.ndarray,
    *,
    reg: float = 0.02,
) -> np.ndarray:
    """Wiener deconvolution stub for BCLP co-optimization inverse step."""
    from numpy.fft import fftn, ifftn

    pad = tuple(k - 1 for k in kernel.shape)
    full = np.pad(signal, [(p // 2, p - p // 2) for p in pad], mode="constant")
    k_pad = np.zeros_like(full)
    off = [(full.shape[i] - kernel.shape[i]) // 2 for i in range(3)]
    k_pad[off[0] : off[0] + kernel.shape[0], off[1] : off[1] + kernel.shape[1], off[2] : off[2] + kernel.shape[2]] = kernel
    f_sig = fftn(full)
    f_ker = fftn(k_pad)
    denom = np.abs(f_ker) ** 2 + reg
    out = np.real(ifftn(f_sig * np.conj(f_ker) / denom))
    sl = tuple(slice(p // 2, p // 2 + signal.shape[i]) for i, p in enumerate(pad))
    return np.maximum(out[sl], 0.0).astype(np.float64)
