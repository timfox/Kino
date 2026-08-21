"""Tomographic FBP / Radon adjoint stubs (Eq. 2–6)."""

from __future__ import annotations

import numpy as np

from ltx_trainer.cal_diffusion.kernel import convolve_volume, gaussian_kernel_3d


def plate_target_volume(shape: tuple[int, int, int]) -> np.ndarray:
    """Sharp plate-like test geometry with square features (Fig. 1)."""
    nx, ny, nz = shape
    vol = np.zeros(shape, dtype=np.float64)
    cx, cy = nx // 2, ny // 2
    half = min(nx, ny) // 3
    vol[cx - half : cx + half, cy - half : cy + half, :] = 1.0
    hole = max(3, min(nx, ny) // 8)
    vol[cx - hole // 2 : cx + hole // 2, cy - hole // 2 : cy + hole // 2, nz // 3 : 2 * nz // 3] = 0.0
    bump = max(2, hole // 2)
    vol[cx + half - bump : cx + half, cy - bump // 2 : cy + bump // 2, :] = 1.0
    return vol


def ct_blur_volume(volume: np.ndarray, *, voxel_m: float) -> np.ndarray:
    """Ramp-filtered back-projection blur proxy."""
    k = gaussian_kernel_3d(volume.shape, d_m2_s=2.5e-11, t_s=80.0, voxel_m=voxel_m, truncate_sigma=2.0)
    return convolve_volume(volume, k)


def blur_operator(
    volume: np.ndarray,
    *,
    d_m2_s: float,
    t_s: float,
    voxel_m: float,
) -> np.ndarray:
    """CT blur then diffusion (Eq. 5)."""
    dose = ct_blur_volume(volume, voxel_m=voxel_m)
    if d_m2_s <= 0.0:
        return dose
    k_diff = gaussian_kernel_3d(volume.shape, d_m2_s=d_m2_s, t_s=t_s, voxel_m=voxel_m)
    return convolve_volume(dose, k_diff)


def forward_dose(
    target: np.ndarray,
    *,
    d_m2_s: float,
    t_s: float,
    voxel_m: float,
    include_ct_blur: bool = True,
) -> np.ndarray:
    """Physical dose deposition."""
    dose = target.astype(np.float64)
    if include_ct_blur:
        return blur_operator(dose, d_m2_s=d_m2_s, t_s=t_s, voxel_m=voxel_m)
    k = gaussian_kernel_3d(target.shape, d_m2_s=d_m2_s, t_s=t_s, voxel_m=voxel_m)
    return convolve_volume(dose, k)


def equal_volume_binarize(dose: np.ndarray, target_volume: int) -> np.ndarray:
    """Threshold dose to match target voxel count (Fig. 2)."""
    flat = dose.ravel()
    if target_volume <= 0:
        return (dose > 0.5).astype(np.float64)
    if target_volume >= flat.size:
        return np.ones_like(dose)
    idx = np.argpartition(flat, -target_volume)[-target_volume]
    thresh = flat[idx]
    return (dose >= thresh).astype(np.float64)
