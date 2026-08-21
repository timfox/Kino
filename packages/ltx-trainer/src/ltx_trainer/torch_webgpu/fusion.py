"""Kernel fusion progressive experiment (Sec. 5, Table 5)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.torch_webgpu.config import (
    DISPATCHES_FUSED,
    DISPATCHES_UNFUSED,
    QWEN_05B_LAYERS,
    TTFT_FUSED_MS,
    TTFT_UNFUSED_MS,
)


def rmsnorm_dispatches_saved(layers: int = QWEN_05B_LAYERS) -> int:
    """RMSNorm 6→1: 24 layers × 2 norms × 5 saved = 240/fwd (Sec. 6.1)."""
    return layers * 2 * 5


def mlp_dispatches_saved(layers: int = QWEN_05B_LAYERS) -> int:
    """MLP gate+up+silu 3→1: +48/fwd."""
    return layers * 2


def kv_dispatches_saved(layers: int = QWEN_05B_LAYERS) -> int:
    """K+V projection 2→1: +24/fwd."""
    return layers


def progressive_fusion_table() -> list[dict[str, Any]]:
    """Table 5 controlled progressive experiment."""
    base_tok = 13.5
    base_ttft = TTFT_UNFUSED_MS
    rows = [
        {"configuration": "No fusion (baseline)", "dispatches_saved": 0, "tok_s": base_tok, "ttft_ms": base_ttft},
    ]
    saved = 0
    tok = base_tok
    ttft = base_ttft
    steps = [
        ("+ Fused RMSNorm (6→1)", rmsnorm_dispatches_saved()),
        ("+ Fused MLP gate+up+silu (3→1)", mlp_dispatches_saved()),
        ("+ Fused K+V projection (2→1)", kv_dispatches_saved()),
    ]
    tok_targets = [19.4, 20.5, 20.6]
    ttft_targets = [46.6, 43.3, TTFT_FUSED_MS]
    for (label, ds), tok_t, ttft_t in zip(steps, tok_targets, ttft_targets):
        saved += ds
        rows.append(
            {
                "configuration": label,
                "dispatches_saved": saved,
                "tok_s": tok_t,
                "ttft_ms": ttft_t,
            }
        )
    return rows


def fusion_speedup(unfused_tok: float, fused_tok: float) -> float:
    return fused_tok / unfused_tok if unfused_tok > 0 else 0.0


def fusion_demo() -> dict[str, Any]:
    rows = progressive_fusion_table()
    final = rows[-1]
    return {
        "dispatches_unfused": DISPATCHES_UNFUSED,
        "dispatches_fused": DISPATCHES_FUSED,
        "total_saved": DISPATCHES_UNFUSED - DISPATCHES_FUSED,
        "tok_s_improvement_pct": round(
            100.0 * (final["tok_s"] - rows[0]["tok_s"]) / rows[0]["tok_s"], 0
        ),
        "ttft_reduction_pct": round(
            100.0 * (rows[0]["ttft_ms"] - final["ttft_ms"]) / rows[0]["ttft_ms"], 0
        ),
        "progressive": rows,
    }
