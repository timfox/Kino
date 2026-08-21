"""First-frame Trellis2 anchor conditioning (Sec. 4.1)."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.helix4d.config import Helix4DConfig


def build_frame_timesteps(num_frames: int, *, anchor_clean: bool = True) -> Tensor:
    """Anchor frame τ=0 (clean); frames f≥1 use standard noisy schedule."""
    t = torch.ones(num_frames)
    if anchor_clean:
        t[0] = 0.0
    return t


def flow_matching_loss_mask(num_frames: int, *, anchor_frame: int = 0) -> Tensor:
    """Train loss only on frames f ≥ 1 (anchor uses GT latent)."""
    mask = torch.ones(num_frames, dtype=torch.bool)
    mask[anchor_frame] = False
    return mask


def inject_anchor_latent(
    latents: Tensor,
    anchor_latent: Tensor,
    *,
    anchor_frame: int = 0,
) -> Tensor:
    """Replace anchor frame slice with clean Trellis2-generated latent."""
    out = latents.clone()
    if latents.dim() == 3:
        out[:, anchor_frame] = anchor_latent
    else:
        out[anchor_frame] = anchor_latent
    return out


def trellis2_stage_order(cfg: Helix4DConfig) -> list[str]:
    """Three flow-matching stages (Fig. 2)."""
    return list(cfg.stages)
