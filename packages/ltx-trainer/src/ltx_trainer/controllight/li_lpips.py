"""LI-LPIPS-style edge-aware consistency proxy (CLE Diffusion / Table 4 ablation)."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.controllight.edges import edge_difference_map, structural_edge_response
from ltx_trainer.controllight.retinex import luminance_y


def color_normalized_flatten(image: Tensor) -> Tensor:
    """Per-image luminance normalization before feature distance."""
    y = luminance_y(image.clamp(0, 1))
    mean = y.mean()
    std = y.std().clamp(min=1e-4)
    norm = (image - mean) / std
    return norm.flatten()


def li_lpips_proxy(a: Tensor, b: Tensor) -> float:
    """Lower is better: L2 on color-normalized pixels + edge-difference penalty."""
    fa = color_normalized_flatten(a)
    fb = color_normalized_flatten(b)
    feat = (fa - fb).pow(2).mean()
    edge = edge_difference_map(a, b).mean()
    return float((feat + 0.5 * edge).sqrt().item())


def lwfm_ablation_smoke(
    i0: Tensor,
    i1: Tensor,
    *,
    misalign_shift_px: int = 2,
) -> dict[str, float]:
    """Synthetic L_FM vs L_wFM: shifted target increases LI-LPIPS; weights reduce effective error."""
    from ltx_trainer.controllight.edges import misalignment_weight_map, resize_weight_to_latent
    from ltx_trainer.controllight.losses import flow_matching_loss

    # Simulate velocity error dominated by misaligned target edges
    shifted = torch.roll(i1, shifts=misalign_shift_px, dims=-1)
    v_pred = torch.randn(1, 4, 16, 16)
    v_star = torch.randn(1, 4, 16, 16)
    li_mis = li_lpips_proxy(i0, shifted)

    w = misalignment_weight_map(i0, shifted)
    f_w = resize_weight_to_latent(w, v_pred.shape)
    while f_w.dim() < v_pred.dim():
        f_w = f_w.unsqueeze(-1)
    f_w = f_w.expand_as(v_pred)
    loss_fm = float(flow_matching_loss(v_pred, v_star).item())
    loss_wfm = float(flow_matching_loss(v_pred, v_star, weight=f_w).item())

    return {
        "LI-LPIPS_misaligned": li_mis,
        "L_FM_loss": loss_fm,
        "LwFM_loss": loss_wfm,
        "LwFM_reduces_loss": float(loss_wfm < loss_fm),
    }
