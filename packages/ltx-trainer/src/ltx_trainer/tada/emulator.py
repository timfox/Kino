"""Symmetric convolution pipeline emulator (Sec. 3.2)."""

from __future__ import annotations

import numpy as np

Kernel3x3 = np.ndarray


def symmetric_kernel_from_params(a: float, b: float, c: float) -> Kernel3x3:
    """Build [b,c,b; c,a,c; b,c,b] with a + 4(b+c) = 1."""
    k = np.array([[b, c, b], [c, a, c], [b, c, b]], dtype=np.float64)
    s = k.sum()
    if abs(s) > 1e-12:
        k = k / s
    return k


def project_to_symmetric_sum_one(kernel: Kernel3x3) -> Kernel3x3:
    """Enforce symmetric 3×3 kernel with weights summing to 1."""
    k = np.asarray(kernel, dtype=np.float64).copy()
    a = float(k[1, 1])
    b = float((k[0, 0] + k[0, 2] + k[2, 0] + k[2, 2]) / 4.0)
    c = float((k[0, 1] + k[1, 0] + k[1, 2] + k[2, 1]) / 4.0)
    return symmetric_kernel_from_params(a, b, c)


def apply_emulator(image: np.ndarray, kernel: Kernel3x3) -> np.ndarray:
    """Convolve RGB or grayscale TIF with pipeline emulator."""
    k = project_to_symmetric_sum_one(kernel)
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 2:
        return _conv2d(img, k)
    out = np.stack([_conv2d(img[..., c], k) for c in range(img.shape[-1])], axis=-1)
    return out


def _conv2d(img: np.ndarray, kernel: Kernel3x3) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view

    padded = np.pad(img, 1, mode="edge")
    patches = sliding_window_view(padded, (3, 3))
    return np.tensordot(patches, kernel, axes=((2, 3), (0, 1)))


# Table 1 reference kernels (toy targets).
DENOISE_KERNEL_ORIGINAL = np.array(
    [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ],
    dtype=np.float64,
)

SHARPEN_KERNEL_ORIGINAL = np.array(
    [
        [0.0, -0.25, 0.0],
        [-0.25, 2.0, -0.25],
        [0.0, -0.25, 0.0],
    ],
    dtype=np.float64,
)

DENOISE_KERNEL_TADA = np.array(
    [
        [0.042, 0.105, 0.042],
        [0.105, 0.41, 0.105],
        [0.042, 0.105, 0.042],
    ],
    dtype=np.float64,
)

SHARPEN_KERNEL_TADA = np.array(
    [
        [0.054, -0.37, 0.054],
        [-0.37, 2.26, -0.37],
        [0.054, -0.37, 0.054],
    ],
    dtype=np.float64,
)
