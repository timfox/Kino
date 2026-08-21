"""Chunk-step AR inference (Algorithm 1) and causal DiT smoke."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.echo_infinity.config import EchoInfinityConfig
from ltx_trainer.echo_infinity.kv_cache import (
    StreamingSessionState,
    append_chunk_to_local,
    concat_attention_sources,
    constant_memory_budget_tokens,
)
from ltx_trainer.echo_infinity.memory import MemoryQueryStack, memory_token_count
from ltx_trainer.echo_infinity.relative_rope import layout_for_step, verify_layout_in_range


class CausalChunkDiT(nn.Module):
    """Minimal causal video DiT head for memory-query smoke (not full Wan backbone)."""

    def __init__(self, dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.in_proj = nn.Linear(dim, hidden)
        self.out_proj = nn.Linear(hidden, dim)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, q_cur: Tensor, k_all: Tensor, v_all: Tensor) -> Tensor:
        """Scaled dot-product over concatenated history + current tokens."""
        q = self.in_proj(q_cur)
        k = self.in_proj(k_all)
        v = self.in_proj(v_all)
        scale = q.shape[-1] ** -0.5
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        h = torch.matmul(attn, v)
        return self.out_proj(self.norm(h))


def echo_chunk_step(
    state: StreamingSessionState,
    k_cur: Tensor,
    v_cur: Tensor,
    memory: MemoryQueryStack,
    dit: CausalChunkDiT,
    cfg: EchoInfinityConfig,
) -> tuple[Tensor, StreamingSessionState, dict[str, Any]]:
    """Single Algorithm-1 iteration: attend → sample → append → maybe update Q."""
    f_star = state.global_frame_index
    layout = layout_for_step(f_star, cfg, has_memory=state.tier.has_history)
    assert verify_layout_in_range(layout, cfg.fmax)

    batch = k_cur.shape[0]
    if state.tier.has_history and cfg.enable_memory_update:
        k_mem, v_mem = memory.project_kv(state.memory_q)
    else:
        k_mem = v_mem = None

    k_all, v_all = concat_attention_sources(
        state.tier.k_sink,
        state.tier.v_sink,
        k_mem,
        v_mem,
        state.tier.k_local,
        state.tier.v_local,
        k_cur,
        v_cur,
    )
    out = dit(k_cur, k_all, v_all)

    k_local, v_local, k_evict, v_evict, evicted = append_chunk_to_local(
        k_cur,
        v_cur,
        state.tier.k_local,
        state.tier.v_local,
        max_frames=cfg.local_window_frames,
        tokens_per_frame=cfg.tokens_per_frame,
    )
    state.tier.k_local = k_local
    state.tier.v_local = v_local
    frames_chunk = k_cur.shape[1] // cfg.tokens_per_frame
    state.global_frame_index += frames_chunk
    state.local_frame_count = (
        0 if k_local is None else k_local.shape[1] // cfg.tokens_per_frame
    )

    if evicted and k_evict is not None and v_evict is not None:
        state.memory_q = memory.update(state.memory_q, k_evict, v_evict, enabled=cfg.enable_memory_update)
        state.tier.has_history = True

    meta = {
        "f_star": f_star,
        "rope_max": layout.max_id,
        "evicted": evicted,
        "memory_tokens": memory_token_count(cfg),
    }
    return out, state, meta


def init_streaming_session(batch: int, cfg: EchoInfinityConfig, device: torch.device | str = "cpu") -> StreamingSessionState:
    memory = MemoryQueryStack(cfg).to(device)
    q = memory.init_queries(batch)
    return StreamingSessionState(memory_q=q)


def rollout_chunks(
    num_chunks: int,
    cfg: EchoInfinityConfig | None = None,
    *,
    batch: int = 1,
    device: str = "cpu",
) -> dict[str, Any]:
    """Smoke rollout for ``num_chunks`` with random K/V."""
    cfg = cfg or EchoInfinityConfig()
    torch.manual_seed(0)
    dim = cfg.hidden_dim
    tpf = cfg.tokens_per_frame
    b_frames = cfg.chunk_size_frames
    t_chunk = b_frames * tpf

    memory = MemoryQueryStack(cfg).to(device)
    dit = CausalChunkDiT(dim).to(device)
    state = StreamingSessionState(memory_q=memory.init_queries(batch).to(device))

    losses: list[float] = []
    for _ in range(num_chunks):
        k_cur = torch.randn(batch, t_chunk, dim, device=device)
        v_cur = torch.randn(batch, t_chunk, dim, device=device)
        out, state, meta = echo_chunk_step(state, k_cur, v_cur, memory, dit, cfg)
        losses.append(float(out.pow(2).mean().detach()))

    budget = constant_memory_budget_tokens(
        cfg.num_sink_frames,
        cfg.local_window_frames,
        cfg.num_memory_query_frames,
        cfg.chunk_size_frames,
        cfg.tokens_per_frame,
    )
    return {
        "num_chunks": num_chunks,
        "final_frame_index": state.global_frame_index,
        "has_history": state.tier.has_history,
        "active_token_budget": budget,
        "loss_tail": losses[-1] if losses else 0.0,
        "rope_ok": True,
    }
