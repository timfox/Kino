"""Egocentric HDR environment map capture (Sec. 8, supplemental Eq. 31)."""

from __future__ import annotations

import numpy as np


def finlayson_ldr_to_hdr(
    ldr_rgb: np.ndarray,
    A: np.ndarray,
    gamma: float = 2.2,
) -> np.ndarray:
    """
    Second-order root-polynomial color correction (Finlayson et al. 2015).

    ldr_rgb: (H, W, 3) linear RGB in [0, 1]
  A: (3, 6) matrix
    """
    r, gch, b = ldr_rgb[..., 0], ldr_rgb[..., 1], ldr_rgb[..., 2]
    gam = max(gamma, 1e-3)
    rp, gp, bp = r**gam, gch**gam, b**gam
    feats = np.stack(
        [
            rp,
            gp,
            bp,
            np.sqrt(np.clip(rp * gp, 0, None)),
            np.sqrt(np.clip(gp * bp, 0, None)),
            np.sqrt(np.clip(bp * rp, 0, None)),
        ],
        axis=-1,
    )  # (H, W, 6)
    hdr = np.einsum("ij,...j->...i", A, feats)
    return np.clip(hdr, 0.0, 1.0).astype(np.float32)


def hdr_regularization(hdr: np.ndarray) -> float:
    """Eq. 32."""
    over = np.maximum(hdr - 1.0, 0.0)
    under = np.maximum(-hdr, 0.0)
    return float((over**2).mean() + (under**2).mean())


def optimize_hdr_color_correction(
    ldr_panorama: np.ndarray,
    target_mean: np.ndarray | None = None,
    *,
    steps: int = 50,
    lr: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Simplified Adam-free optimization of A (3x6) and gamma for smoke tests.
    """
    A = np.eye(3, 6, dtype=np.float32)
    A[:, :3] = np.eye(3) * 0.9
    gamma = 2.2
    tgt = target_mean if target_mean is not None else ldr_panorama.mean(axis=(0, 1))
    best_loss = float("inf")
    best_A, best_g = A.copy(), gamma
    for _ in range(steps):
        hdr = finlayson_ldr_to_hdr(ldr_panorama, A, gamma)
        loss = float(np.mean((hdr.mean(axis=(0, 1)) - tgt) ** 2)) + hdr_regularization(hdr)
        if loss < best_loss:
            best_loss, best_A, best_g = loss, A.copy(), gamma
        # finite-diff style nudge on diagonal
        for c in range(3):
            A[c, c] = np.clip(A[c, c] + lr * (tgt[c] - hdr[..., c].mean()), 0.1, 1.5)
        gamma = float(np.clip(gamma + lr * 0.01 * (tgt.mean() - hdr.mean()), 1.8, 2.4))
        A, gamma = best_A, best_g
    return best_A, np.array([best_g, best_g, best_g], dtype=np.float32), best_loss
