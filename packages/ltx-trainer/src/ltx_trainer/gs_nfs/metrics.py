"""Quantization, RLGR block proxy, frame metrics."""

from __future__ import annotations

import numpy as np


def dead_zone_quantize(x: np.ndarray, *, step: float) -> np.ndarray:
    return np.round(x / step) * step


def rlgr_size_proxy(coeffs: np.ndarray, *, block_size: int) -> int:
    """Block-parallel RLGR size proxy: run-length on zeros per block."""
    flat = coeffs.reshape(-1)
    total = 0
    for start in range(0, len(flat), block_size):
        block = flat[start : start + block_size]
        zeros = 0
        for v in block:
            if abs(v) < 1e-6:
                zeros += 1
            else:
                total += max(1, zeros // 4) + 8
                zeros = 0
        total += max(1, zeros // 4)
    return int(total)


def psnr_proxy(original: np.ndarray, reconstructed: np.ndarray) -> float:
    mse = float(np.mean((original - reconstructed) ** 2))
    if mse < 1e-12:
        return 99.0
    peak = float(np.max(original) ** 2) if np.max(np.abs(original)) > 1 else 1.0
    return 10.0 * np.log10(peak / mse)


def frame_size_mb(n_gaussians: int, *, bytes_per_gaussian: int = 236) -> float:
    return n_gaussians * bytes_per_gaussian / (1024.0 * 1024.0)


def compressed_mb(raw_mb: float, *, ratio: float) -> float:
    return raw_mb / ratio
