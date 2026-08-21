"""No-reference IQA + PSQ stubs (Sec. 4.3.1)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def niqe_stub(rgb: NDArray[np.floating]) -> float:
    """Lower is better; variance-based naturalness proxy."""
    g = rgb.mean(axis=-1) if rgb.ndim == 3 else rgb
    return float(3.0 + 0.5 * g.std())


def brisque_stub(rgb: NDArray[np.floating]) -> float:
    gx = np.abs(np.diff(rgb.mean(axis=-1), axis=1)).mean() if rgb.ndim == 3 else np.abs(np.diff(rgb, axis=1)).mean()
    return float(20.0 + 30.0 * gx)


def piqe_stub(rgb: NDArray[np.floating]) -> float:
    return float(11.0 + 2.0 * niqe_stub(rgb))


def psq_stub(seam: NDArray[np.floating], saliency: NDArray[np.floating]) -> float:
    """Perceptual Seam Quality in [0,1]; lower is better."""
    edge = np.abs(np.diff(seam, axis=1)).mean() + np.abs(np.diff(seam, axis=0)).mean()
    return float(np.clip(0.05 + edge * saliency.mean(), 0, 1))


def metric_bundle(rgb: NDArray[np.floating], seam: NDArray[np.floating], saliency: NDArray[np.floating]) -> dict[str, float]:
    return {
        "NiQE": niqe_stub(rgb),
        "BRISQUE": brisque_stub(rgb),
        "PIQE": piqe_stub(rgb),
        "PSQ": psq_stub(seam, saliency),
    }
