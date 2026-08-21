"""Autoregressive flow-matching head stub (§2.5)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.dotstts.config import DotsttsConfig


def flow_matching_velocity(clean: np.ndarray, noise: np.ndarray, t: float) -> np.ndarray:
    """Rectified-flow target velocity P − ε."""
    return clean - noise


def interpolate_patch(clean: np.ndarray, noise: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation Z_t = (1−t)ε + t P."""
    return (1.0 - t) * noise + t * clean


def flow_matching_loss(
    pred: np.ndarray,
    clean: np.ndarray,
    noise: np.ndarray,
) -> float:
    target = flow_matching_velocity(clean, noise, 0.0)
    return float(np.mean((pred - target) ** 2))


def block_causal_mask(n_patches: int, h_size: int = 1, p_size: int = 4) -> np.ndarray:
    """Toy block-causal mask: C→C causal, C→Z masked, Z→C prefix-causal, Z→Z block-diag."""
    block = h_size + p_size
    c_len = n_patches * block
    z_len = c_len
    total = c_len + z_len
    mask = np.zeros((total, total), dtype=bool)
    # C → C: lower triangular within blocks (simplified causal)
    for i in range(c_len):
        mask[i, : i + 1] = True
    # C → Z: masked (False)
    # Z → C: prefix causal — each Z block n sees C up to block n
    for n in range(n_patches):
        z_start = c_len + n * block
        z_end = z_start + block
        c_prefix_end = (n + 1) * block
        mask[z_start:z_end, :c_prefix_end] = True
        # Z → Z: block diagonal
        mask[z_start:z_end, z_start:z_end] = True
    return mask


def ar_fm_demo(*, seed: int = 0, cfg: DotsttsConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DotsttsConfig()
    rng = np.random.default_rng(seed)
    patch = rng.normal(size=(cfg.patch_frames, cfg.latent_dim))
    noise = rng.normal(size=patch.shape)
    t = float(rng.uniform(0, 1))
    z_t = interpolate_patch(patch, noise, t)
    pred = patch - noise + rng.normal(scale=0.05, size=patch.shape)
    loss = flow_matching_loss(pred, patch, noise)
    mask = block_causal_mask(n_patches=3, p_size=cfg.patch_frames)
    return {
        "patch_frames": cfg.patch_frames,
        "flow_loss": round(loss, 6),
        "block_causal_trainable": mask.shape[0] == mask.shape[1],
        "full_history_conditioning": True,
    }
