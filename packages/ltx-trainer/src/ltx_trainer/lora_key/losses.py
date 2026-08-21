"""Stage 2 watermark LoRA objectives — Eqs. 10–12."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def watermark_consistency_loss(eps_key: Tensor, eps_base: Tensor) -> Tensor:
    """L_wm = ‖ε̂_key − ε̂_base‖² (Eq. 10)."""
    return F.mse_loss(eps_key, eps_base)


def semantic_cosine_loss(ref_feat: Tensor, wm_feat: Tensor) -> Tensor:
    """L_sem = 1 − cos(F_ref, F_wm) (Eq. 12)."""
    ref = ref_feat.detach()
    cos = F.cosine_similarity(ref.flatten(1), wm_feat.flatten(1), dim=1)
    return (1.0 - cos).mean()


def estimate_z0(z_t: Tensor, eps_pred: Tensor, alpha_bar_t: float) -> Tensor:
    """ẑ0 from deterministic scheduler (Eq. 11)."""
    return z_t - (1.0 - alpha_bar_t) ** 0.5 * eps_pred / (alpha_bar_t**0.5 + 1e-8)
