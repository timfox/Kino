"""Cross-frame attention with first-frame anchor (Sec. 4.1, Eq. 1–2)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.helix4d.config import Helix4DConfig


def frame_token_offsets(tokens_per_frame: list[int] | Tensor) -> Tensor:
    """Cumulative start index per frame for flattened token layout."""
    if isinstance(tokens_per_frame, Tensor):
        counts = tokens_per_frame.tolist()
    else:
        counts = list(tokens_per_frame)
    offsets = [0]
    for c in counts:
        offsets.append(offsets[-1] + c)
    return torch.tensor(offsets, dtype=torch.long)


def sliding_window_anchor_mask(
    num_frames: int,
    tokens_per_frame: int | list[int] | Tensor,
    *,
    window_half: int,
    anchor_frame: int = 0,
) -> Tensor:
    """Binary mask M ∈ {0,1}^{S×S} per Eq. (2)."""
    if isinstance(tokens_per_frame, int):
        counts = [tokens_per_frame] * num_frames
    elif isinstance(tokens_per_frame, Tensor):
        counts = tokens_per_frame.tolist()
    else:
        counts = list(tokens_per_frame)
    total = sum(counts)
    m = torch.zeros(total, total)
    offsets = frame_token_offsets(counts)
    for f in range(num_frames):
        q0, q1 = int(offsets[f]), int(offsets[f + 1])
        for fp in range(num_frames):
            in_window = abs(f - fp) <= window_half
            is_anchor = anchor_frame >= 0 and fp == anchor_frame
            if is_anchor or in_window:
                k0, k1 = int(offsets[fp]), int(offsets[fp + 1])
                m[q0:q1, k0:k1] = 1.0
    return m


def masked_attention_logits(
    queries: Tensor,
    keys: Tensor,
    mask: Tensor,
    *,
    scale: float | None = None,
) -> Tensor:
    """Eq. (1): exp(q·k/√d) with mask, normalized over keys."""
    d = queries.shape[-1]
    scale = scale if scale is not None else d**-0.5
    logits = torch.matmul(queries, keys.transpose(-2, -1)) * scale
    if mask.dim() == 2:
        logits = logits.masked_fill(mask == 0, float("-inf"))
    attn = F.softmax(logits, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)
    return attn


def apply_cross_frame_attention(
    features: Tensor,
    mask: Tensor,
    *,
    num_heads: int = 4,
) -> Tensor:
    """Single-head smoke: project and aggregate with masked attention."""
    b, s, d = features.shape
    q = k = v = features
    attn = masked_attention_logits(q, k, mask.to(features.device))
    return torch.matmul(attn, v)


def mask_pattern_name(cfg: Helix4DConfig, pattern: str) -> Tensor:
    """Build masks for Tab. 4 ablation patterns."""
    f = cfg.num_frames
    s_pf = 8
    w = cfg.attention_window // 2
    if pattern == "full":
        n = f * s_pf
        return torch.ones(n, n)
    if pattern == "causal":
        m = torch.zeros(f * s_pf, f * s_pf)
        offs = frame_token_offsets([s_pf] * f)
        for fr in range(f):
            for fp in range(fr + 1):
                m[offs[fr] : offs[fr + 1], offs[fp] : offs[fp + 1]] = 1.0
        return m
    if pattern == "sliding":
        return sliding_window_anchor_mask(f, s_pf, window_half=w, anchor_frame=-1)  # no anchor
    if pattern == "spatial":
        m = torch.zeros(f * s_pf, f * s_pf)
        offs = frame_token_offsets([s_pf] * f)
        block = s_pf // 8
        for fr in range(f):
            for fp in range(f):
                if fr // block == fp // block:
                    m[offs[fr] : offs[fr + 1], offs[fp] : offs[fp + 1]] = 1.0
        return m
    if pattern == "ours":
        return sliding_window_anchor_mask(f, s_pf, window_half=w, anchor_frame=0)
    raise ValueError(f"unknown pattern: {pattern}")
