"""Multi-gate dense fusion with SE residual gating (Eq. 5–6)."""

from __future__ import annotations

import numpy as np


def _global_avg_pool(feat: np.ndarray) -> np.ndarray:
    """GAP over spatial dims; feat is C×H×W."""
    return feat.mean(axis=(1, 2))


def se_residual_gate(fuse: np.ndarray, *, squeeze_ratio: int = 4) -> np.ndarray:
    """SE block: F_out = F_fuse + F_fuse ⊙ σ(FC(ReLU(FC(GAP)))) (Eq. 6)."""
    c = fuse.shape[0]
    hidden = max(1, c // squeeze_ratio)
    s = _global_avg_pool(fuse)
    # Two-layer MLP stub with fixed pseudo-weights from channel statistics.
    w1 = np.linspace(0.5, 1.5, hidden * c).reshape(hidden, c) / c
    w2 = np.linspace(1.0, 0.5, c * hidden).reshape(c, hidden) / hidden
    h = np.maximum(0.0, w1 @ s)
    scale = 1.0 / (1.0 + np.exp(-(w2 @ h)))
    return fuse + fuse * scale[:, np.newaxis, np.newaxis]


def bilinear_align_to(
    feat: np.ndarray,
    target_hw: tuple[int, int],
) -> np.ndarray:
    """Resize C×H×W feature map to target spatial size (numpy nearest for stub)."""
    c, h, w = feat.shape
    th, tw = target_hw
    if (h, w) == (th, tw):
        return feat
    # Simple repeat/interpolate for reference (not training-grade).
    row_idx = (np.linspace(0, h - 1, th)).astype(int)
    col_idx = (np.linspace(0, w - 1, tw)).astype(int)
    return feat[:, row_idx, :][:, :, col_idx]


def mgf_fuse(
    features: list[np.ndarray],
    *,
    fuse_channels: int | None = None,
) -> np.ndarray:
    """Dense fusion of F3,F4,F5: project, align, concat, residual block + SE (Eq. 5–6)."""
    if len(features) < 2:
        raise ValueError("mgf_fuse expects at least two stage features")
    target_hw = features[-1].shape[1:]
    aligned: list[np.ndarray] = []
    for f in features:
        c = f.shape[0]
        fc = fuse_channels or c
        # 1×1 projection stub: channel-wise scale.
        proj = f * (np.linspace(0.9, 1.1, c)[:, np.newaxis, np.newaxis])
        if proj.shape[0] > fc:
            proj = proj[:fc]
        elif proj.shape[0] < fc:
            pad = np.zeros((fc - proj.shape[0], *proj.shape[1:]), dtype=proj.dtype)
            proj = np.vstack([proj, pad])
        aligned.append(bilinear_align_to(proj, target_hw))
    concat = np.concatenate(aligned, axis=0)
    # Residual block stub: identity + small laplacian-style mix.
    lap = (
        concat
        - 0.25
        * (
            np.roll(concat, 1, axis=1)
            + np.roll(concat, -1, axis=1)
            + np.roll(concat, 1, axis=2)
            + np.roll(concat, -1, axis=2)
        )
    )
    fused = 0.5 * concat + 0.5 * lap
    return se_residual_gate(fused)
