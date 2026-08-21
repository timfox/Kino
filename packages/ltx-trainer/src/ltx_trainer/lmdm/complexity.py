"""Inference complexity classes for LMDM vs LMM (Sec. 4.1)."""

from __future__ import annotations


def lmm_decode_ops(context_s: int, block_o: int, decode_per_frame: int = 1) -> int:
    """O(E_{1:s} + sum_{t=s}^{s+o} D_t) toy count."""
    encode = context_s
    decode = block_o * decode_per_frame
    return encode + decode


def lmdm_enc_dec_ops(context_s: int, block_o: int, steps_k: int, denoise_per_step: int = 1) -> int:
    """O(E_{1:s} + D_{s:T} · K) — noise-wise KV cache."""
    return context_s + block_o * steps_k * denoise_per_step


def lmdm_block_causal_ops(
    context_s: int,
    block_o: int,
    steps_k: int,
    *,
    denoise_per_step: int = 1,
) -> int:
    """O(E_{s-o:s} + D_{s:T} · K) — temporal + noise KV cache after warmup."""
    if block_o <= 0:
        raise ValueError("block_o must be positive")
    encode = min(block_o, context_s)
    return encode + block_o * steps_k * denoise_per_step
