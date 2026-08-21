"""Three-tier KV cache: sink + local window + evicted history (§3.2)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class TieredKVState:
    """Per-layer KV stored pre-RoPE (§B); layer-shared memory queries live outside."""

    k_sink: Tensor | None = None
    v_sink: Tensor | None = None
    k_local: Tensor | None = None
    v_local: Tensor | None = None
    has_history: bool = False

    def active_local_frames(self) -> int:
        if self.k_local is None:
            return 0
        # Caller tracks frame count alongside tensors.
        return getattr(self, "_local_frames", 0)


@dataclass
class StreamingSessionState:
    """Cross-layer streaming state for Algorithm 1."""

    memory_q: Tensor
    tier: TieredKVState = field(default_factory=TieredKVState)
    global_frame_index: int = 0
    local_frame_count: int = 0


def append_chunk_to_local(
    k_new: Tensor,
    v_new: Tensor,
    k_local: Tensor | None,
    v_local: Tensor | None,
    *,
    max_frames: int,
    tokens_per_frame: int,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None, bool]:
    """Append chunk K/V to local window; return evicted (K,V) if window overflows."""
    b, t_new, d = k_new.shape
    if t_new % tokens_per_frame != 0:
        raise ValueError("k_new token count must be multiple of tokens_per_frame")
    frames_new = t_new // tokens_per_frame

    if k_local is None:
        return k_new, v_new, None, None, False

    k_cat = torch.cat([k_local, k_new], dim=1)
    v_cat = torch.cat([v_local, v_new], dim=1)
    max_tokens = max_frames * tokens_per_frame
    if k_cat.shape[1] <= max_tokens:
        return k_cat, v_cat, None, None, False

    evict_tokens = k_cat.shape[1] - max_tokens
    k_evict = k_cat[:, :evict_tokens, :]
    v_evict = v_cat[:, :evict_tokens, :]
    k_keep = k_cat[:, evict_tokens:, :]
    v_keep = v_cat[:, evict_tokens:, :]
    return k_keep, v_keep, k_evict, v_evict, True


def concat_attention_sources(
    k_sink: Tensor | None,
    v_sink: Tensor | None,
    k_mem: Tensor | None,
    v_mem: Tensor | None,
    k_local: Tensor | None,
    v_local: Tensor | None,
    k_cur: Tensor,
    v_cur: Tensor,
) -> tuple[Tensor, Tensor]:
    """Kl_all = [Kl_sink; KQ; Kl_local; Kl_cur] (Algorithm 1)."""
    keys: list[Tensor] = []
    vals: list[Tensor] = []
    for k, v in (
        (k_sink, v_sink),
        (k_mem, v_mem),
        (k_local, v_local),
        (k_cur, v_cur),
    ):
        if k is not None and v is not None and k.numel() > 0:
            keys.append(k)
            vals.append(v)
    if not keys:
        return k_cur, v_cur
    return torch.cat(keys, dim=1), torch.cat(vals, dim=1)


def constant_memory_budget_tokens(
    num_sink: int,
    num_local: int,
    num_memory_frames: int,
    chunk_frames: int,
    tokens_per_frame: int,
) -> int:
    """O(NS·S + NW·S + NQ·S + B·S) — constant in sequence length T."""
    return (num_sink + num_local + num_memory_frames + chunk_frames) * tokens_per_frame
