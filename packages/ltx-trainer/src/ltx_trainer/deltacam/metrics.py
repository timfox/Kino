"""Evaluation metrics including wCLIP-5 (Eq. 6)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def wclip(
    gen_feats: Tensor,
    ref_feats: Tensor,
    *,
    window: int = 2,
) -> float:
    """Windowed cosine similarity (Eq. 6); ``window=2`` → wCLIP-5 (±2 frames)."""
    if gen_feats.shape != ref_feats.shape:
        raise ValueError(f"Shape mismatch {gen_feats.shape} vs {ref_feats.shape}")
    t_len = gen_feats.shape[0]
    g = F.normalize(gen_feats, dim=-1)
    r = F.normalize(ref_feats, dim=-1)
    sims: list[float] = []
    for t in range(t_len):
        best = -1.0
        for k in range(-window, window + 1):
            j = t + k
            if 0 <= j < t_len:
                best = max(best, float((g[t] * r[j]).sum().item()))
        sims.append(best)
    return sum(sims) / max(len(sims), 1)


def wclip5(gen_feats: Tensor, ref_feats: Tensor, *, window: int = 2) -> float:
    """Eq. (6); ``window=2`` is the paper's wCLIP-5 (±2 frames)."""
    return wclip(gen_feats, ref_feats, window=window)


def wclip1(gen_feats: Tensor, ref_feats: Tensor) -> float:
    return wclip(gen_feats, ref_feats, window=1)


def wclip10(gen_feats: Tensor, ref_feats: Tensor) -> float:
    return wclip(gen_feats, ref_feats, window=10)


def table_wclip_sensitivity() -> dict[str, dict[str, float]]:
    """Table 4(b) — wCLIP-1 / wCLIP-5 / wCLIP-10 sensitivity."""
    return {
        "VACE": {"wclip1": 0.85, "wclip5": 0.87, "wclip10": 0.87},
        "CogVideoX": {"wclip1": 0.89, "wclip5": 0.91, "wclip10": 0.92},
        "Ours": {"wclip1": 0.94, "wclip5": 0.95, "wclip10": 0.95},
    }


def reference_metrics_placeholder(
    mse: float,
    *,
    psnr_scale: float = 30.0,
) -> dict[str, float]:
    """Proxy PSNR/SSIM/LPIPS from MSE for smoke tests (not real perceptual metrics)."""
    psnr = psnr_scale - 10.0 * mse
    ssim = max(0.0, 1.0 - mse)
    lpips = min(1.0, mse * 2.0)
    return {"psnr": psnr, "ssim": ssim, "lpips": lpips}
