"""CUDA block mapping cards (§3) — registers, shared memory, scans, variant mult."""

from __future__ import annotations

from typing import Any

from ltx_trainer.midint_division.constants import (
    DEFAULT_SEQUENTIALIZATION_Q,
    DEFAULT_WORD_BITS,
    REGISTER_SPILL_BYTES_2P18,
    SHARED_MEMORY_LIMIT_KB,
)


def digits_m(bits_exp: int, word_bits: int = DEFAULT_WORD_BITS) -> int:
    """Digit count M for 2^bits_exp total bits (uint word size 16/32/64)."""
    return (1 << bits_exp) // word_bits


def block_layout_card(bits_exp: int, q: int = DEFAULT_SEQUENTIALIZATION_Q) -> dict[str, Any]:
    """One division instance → one CUDA block; M = Q · blockDim.x."""
    word_bits = DEFAULT_WORD_BITS
    total_bits = 1 << bits_exp
    m = total_bits // word_bits
    block_dim = m // q
    shared_bytes = 2 * m * (word_bits // 8)  # classical mul manifests both operands
    return {
        "bits": total_bits,
        "bits_exp": bits_exp,
        "word_bits": word_bits,
        "digits_m": m,
        "sequentialization_q": q,
        "block_dim_x": block_dim,
        "shared_memory_kb": round(shared_bytes / 1024, 1),
        "shared_limit_kb": SHARED_MEMORY_LIMIT_KB,
        "fits_shared": shared_bytes <= SHARED_MEMORY_LIMIT_KB * 1024,
        "register_spill_bytes_2p18": REGISTER_SPILL_BYTES_2P18,
    }


def operation_catalog() -> list[dict[str, str]]:
    return [
        {"op": "cpyGlb2Reg", "memory": "global→shared→register", "listing": "1.1"},
        {"op": "shift", "memory": "register + shared staging", "listing": "1.2"},
        {"op": "subPowB", "memory": "register + atomicMin", "listing": "1.3"},
        {"op": "lt", "memory": "register + scanBlk LTop", "listing": "1.4"},
        {"op": "subRegs", "memory": "register + CarryOP scan", "listing": "1.5–1.6"},
        {"op": "effMul", "memory": "shared + q=4 schedule", "listing": "§3.4 / Fig. 2"},
    ]


def variant_mult_dispatch() -> list[str]:
    return [
        "if (m <= BLOCK) smallMul<uint>(m, ...)",
        "else if (m <= 2*BLOCK) effMul<uint, BLOCK, 2>(...)",
        "else if (m <= 4*BLOCK) effMul<uint, BLOCK, 4>(...)",
        "else if (m <= 8*BLOCK) effMul<uint, BLOCK, 8>(...)",
    ]
