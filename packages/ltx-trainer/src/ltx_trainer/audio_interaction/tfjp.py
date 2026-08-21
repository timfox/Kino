"""Time-frequency joint preprocessing stub (Algorithm 1)."""

from __future__ import annotations

import torch


def silence_cut(x: torch.Tensor, silence_limit_ratio: float = 0.15) -> torch.Tensor:
    """Trim low-energy runs at segment edges (stub)."""
    if x.numel() == 0:
        return x
    flat = x.reshape(-1, x.shape[-1])
    out = []
    for row in flat:
        frame_energy = row.abs()
        thresh = torch.quantile(frame_energy, silence_limit_ratio)
        mask = frame_energy >= thresh
        if not mask.any():
            out.append(row)
            continue
        idx = torch.where(mask)[0]
        out.append(row[idx[0] : idx[-1] + 1])
    trimmed = torch.stack(out, dim=0)
    if x.dim() == 1:
        return trimmed.squeeze(0)
    return trimmed.reshape(*x.shape[:-1], trimmed.shape[-1])


def tfjp_preprocess(
    waveform: torch.Tensor,
    *,
    max_iters: int = 3,
) -> torch.Tensor:
    """Lightweight TFJP pipeline for stitching-ready clips."""
    x = waveform.clone()
    for _ in range(max_iters):
        x = silence_cut(x)
        n = x.abs().mean()
        x = x - 0.1 * n
    return silence_cut(x)
