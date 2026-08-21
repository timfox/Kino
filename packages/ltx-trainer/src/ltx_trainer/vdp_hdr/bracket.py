"""Exposure bracket synthesis (Talegaonkar et al. Sec. 4, eq. 6–7)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class BracketConfig:
    num_frames: int = 5
    gamma: float = 2.2
    target_low: float = 0.85
    target_high: float = 0.85
    ev_span: float = 5.0


def _luminance(rgb: Tensor) -> Tensor:
    """BT.709 luminance; supports ``[B,C,H,W]`` or ``[C,H,W]``."""
    if rgb.ndim == 3:
        r, g, b = rgb[0], rgb[1], rgb[2]
    else:
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def exposure_times_from_hdr(hdr_bchw: Tensor, cfg: BracketConfig) -> Tensor:
    """Per-batch exposure times ``t`` shape ``[B, N]`` (paper eq. 7)."""
    if hdr_bchw.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(hdr_bchw.shape)}")
    b = hdr_bchw.shape[0]
    device, dtype = hdr_bchw.device, hdr_bchw.dtype
    y = _luminance(hdr_bchw.clamp(min=0.0))
    y_max = y.amax(dim=(1, 2))
    y_med = y.flatten(1).median(dim=1).values.clamp(min=1e-8)
    g = cfg.gamma
    ts = (cfg.target_low**g) / y_max.clamp(min=1e-8)
    te = (cfg.target_high**g) / y_med
    log2_ts = torch.log2(ts.clamp(min=1e-12))
    log2_te = torch.log2(te.clamp(min=1e-12))
    n = cfg.num_frames
    grid = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
    log2_t = log2_ts.unsqueeze(1) + grid.unsqueeze(0) * (log2_te - log2_ts).unsqueeze(1)
    return (2.0**log2_t).clamp(min=1e-12)


def hdr_to_ldr_bracket(hdr_bchw: Tensor, cfg: BracketConfig | None = None) -> tuple[Tensor, Tensor]:
    """GT LDR bracket ``[B,N,C,H,W]`` in display γ and exposure times ``[B,N]``."""
    cfg = cfg or BracketConfig()
    if hdr_bchw.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(hdr_bchw.shape)}")
    t = exposure_times_from_hdr(hdr_bchw, cfg)
    b, _, h, w = hdr_bchw.shape
    n = cfg.num_frames
    outs: list[Tensor] = []
    inv_g = 1.0 / cfg.gamma
    for i in range(n):
        ti = t[:, i].view(b, 1, 1, 1)
        lin = (hdr_bchw * ti).clamp(0.0, 1.0)
        outs.append(lin.pow(inv_g))
    return torch.stack(outs, dim=1), t


def ldr_bracket_from_single(
    ldr_chw: Tensor,
    cfg: BracketConfig | None = None,
) -> Tensor:
    """Photometric bracket prior ``[1,N,C,H,W]`` from one γ-encoded LDR (inference without VDM)."""
    cfg = cfg or BracketConfig()
    if ldr_chw.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got {tuple(ldr_chw.shape)}")
    sdr = ldr_chw.detach().float().clamp(0.0, 1.0)
    linear = sdr.pow(cfg.gamma)
    half = cfg.ev_span * 0.5
    n = cfg.num_frames
    evs = [half * (2.0 * i / max(n - 1, 1) - 1.0) for i in range(n)]
    outs: list[Tensor] = []
    inv_g = 1.0 / cfg.gamma
    for ev in evs:
        scaled = (linear * (2.0**ev)).clamp(0.0, 1.0)
        outs.append(scaled.pow(inv_g))
    return torch.stack(outs, dim=0).unsqueeze(0)
