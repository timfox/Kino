"""Video-to-video conditioning layout (Sec. 3.1.1)."""

from __future__ import annotations

import numpy as np


def build_frame_mask(
    num_cond_frames: int,
    num_total_frames: int,
    *,
    latent_h: int = 8,
    latent_w: int = 8,
) -> np.ndarray:
    """Binary mask M: 1 = retain conditioning frames, 0 = synthesize."""
    mask = np.zeros((num_total_frames, latent_h, latent_w), dtype=np.float32)
    mask[:num_cond_frames] = 1.0
    return mask


def concat_conditioning_channels(
    noise_latent: np.ndarray,
    condition_latent: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Channel-wise concat of z_t, z_c, and mask for Wan DiT input (toy layout)."""
    mask_arr = np.asarray(mask, dtype=np.float32)
    if mask_arr.ndim == 3:
        # (T, h, w) -> single-channel spatial mask plane
        mask_plane = mask_arr.max(axis=0, keepdims=True)
    elif mask_arr.ndim == 4:
        mask_plane = mask_arr.max(axis=(0, 1), keepdims=True)
        if mask_plane.shape[0] != 1:
            mask_plane = mask_plane[:1]
    else:
        mask_plane = mask_arr.reshape(1, *mask_arr.shape[-2:])
    return np.concatenate([noise_latent, condition_latent, mask_plane], axis=0)
