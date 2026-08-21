"""Dataset variance spectrum S0(k) and power-law fits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ltx_trainer.skild.dct import dct2, frequency_radius_grid


@dataclass
class PowerLawSpectrumFit:
    C: float
    k0_sq: float
    exponent: float

    def evaluate(self, k_radius: np.ndarray) -> np.ndarray:
        return self.C * np.power(k_radius * k_radius + self.k0_sq, -self.exponent)


def radial_variance_spectrum(images: list[np.ndarray], *, max_bins: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Radially averaged variance per frequency bin from DCT coefficients.

    Returns:
        ``(bin_centers, variance)`` with length <= ``max_bins``.
    """
    if not images:
        raise ValueError("Need at least one image")
    h, w = images[0].shape[:2]
    k_grid = frequency_radius_grid(h, w)
    k_max = float(k_grid.max()) + 1e-8
    bins = np.linspace(0.0, k_max, max_bins + 1)
    acc = np.zeros(max_bins, dtype=np.float64)
    cnt = np.zeros(max_bins, dtype=np.float64)

    for img in images:
        x = img.astype(np.float64)
        if x.ndim == 3:
            x = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
        coeff = dct2(x)
        flat_k = k_grid.ravel()
        flat_x = coeff.ravel()
        idx = np.digitize(flat_k, bins) - 1
        idx = np.clip(idx, 0, max_bins - 1)
        for b in range(max_bins):
            mask = idx == b
            if not np.any(mask):
                continue
            vals = flat_x[mask]
            acc[b] += float(np.var(vals))
            cnt[b] += 1.0

    var = np.where(cnt > 0, acc / np.maximum(cnt, 1.0), 0.0)
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, var


def fit_power_law_spectrum(
    bin_centers: np.ndarray,
    variance: np.ndarray,
    *,
    k0_sq: float = 3.0,
) -> PowerLawSpectrumFit:
    """Fit ``S(k) = C (k² + k₀²)^(-a)`` on positive-variance bins."""
    mask = (bin_centers > 0) & (variance > 1e-12)
    if mask.sum() < 3:
        return PowerLawSpectrumFit(C=1.0, k0_sq=k0_sq, exponent=1.0)
    k = bin_centers[mask]
    v = variance[mask]
    # log-linear: log v = log C - a log(k² + k0²)
    x = np.log(k * k + k0_sq)
    y = np.log(v)
    a, log_c = np.polyfit(x, y, 1)
    return PowerLawSpectrumFit(C=float(np.exp(log_c)), k0_sq=k0_sq, exponent=float(-a))


def spectrum_map_from_fit(h: int, w: int, fit: PowerLawSpectrumFit) -> np.ndarray:
    """Per-mode variance map ``S0`` shape ``(H, W)``."""
    k = frequency_radius_grid(h, w)
    k_eff = np.maximum(k, 1e-6)
    return fit.evaluate(k_eff).astype(np.float32)


def load_images_from_dir(path: Path, *, limit: int = 256) -> list[np.ndarray]:
    import cv2  # noqa: PLC0415

    paths = sorted(
        [p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    )[:limit]
    out: list[np.ndarray] = []
    for p in paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        out.append(rgb)
    return out
