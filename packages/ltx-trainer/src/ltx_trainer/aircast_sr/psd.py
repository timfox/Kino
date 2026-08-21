"""Radial power spectral density for perception–distortion analysis."""

from __future__ import annotations

import numpy as np


def radial_power_spectral_density(field: np.ndarray, *, dx_km: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """2D field → (wavelength_km[], mean_power[])."""
    arr = np.asarray(field, dtype=np.float64)
    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[-2], arr.shape[-1])
    f = np.fft.fft2(arr - arr.mean())
    power = (np.abs(np.fft.fftshift(f)) ** 2) / arr.size
    cy, cx = arr.shape[0] // 2, arr.shape[1] // 2
    yy, xx = np.indices(arr.shape)
    r_pix = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = int(r_pix.max())
    bins = np.arange(0.5, r_max + 0.5, 1.0)
    radial = np.zeros(len(bins), dtype=np.float64)
    for i, r in enumerate(bins):
        mask = (r_pix >= r - 0.5) & (r_pix < r + 0.5)
        if mask.any():
            radial[i] = power[mask].mean()
    freq = np.fft.fftfreq(arr.shape[0], d=dx_km)
    freq2d = np.sqrt(np.fft.fftfreq(arr.shape[0], d=dx_km)[:, None] ** 2 + np.fft.fftfreq(arr.shape[1], d=dx_km)[None, :] ** 2)
    freq2d = np.fft.fftshift(freq2d)
    with np.errstate(divide="ignore"):
        wavelength_km = np.where(freq2d > 0, 1.0 / freq2d, np.nan)
    wl_bins = np.zeros(len(bins))
    for i, r in enumerate(bins):
        mask = (r_pix >= r - 0.5) & (r_pix < r + 0.5)
        if mask.any():
            wl_bins[i] = np.nanmedian(wavelength_km[mask])
    valid = np.isfinite(wl_bins) & (wl_bins > 0)
    return wl_bins[valid], radial[valid]
