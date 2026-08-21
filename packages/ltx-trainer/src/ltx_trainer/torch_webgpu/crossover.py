"""Dispatch-bound crossover analysis (Appendix F, Table 14)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.torch_webgpu.config import (
    PER_OPERATION_OVERHEAD_US,
    QWEN_05B_HIDDEN,
    QWEN_05B_INTER,
    QWEN_15B_HIDDEN,
    QWEN_15B_INTER,
    WGSL_THROUGHPUT_TFLOPS,
)


def crossover_batch_size(
    din: int,
    dout: int,
    *,
    overhead_us: float = PER_OPERATION_OVERHEAD_US,
    throughput_tflops: float = WGSL_THROUGHPUT_TFLOPS,
) -> float:
    """B* = T_overhead · throughput / (2 · din · dout) (Appendix F)."""
    denom = 2.0 * din * dout
    if denom <= 0:
        return float("inf")
    return (overhead_us * 1e-6 * throughput_tflops * 1e12) / denom


def crossover_table() -> list[dict[str, Any]]:
    """Table 14 representative operations."""
    specs = [
        ("Qwen2.5-0.5B attention Q/K/V", QWEN_05B_HIDDEN, QWEN_05B_HIDDEN),
        ("Qwen2.5-0.5B MLP up", QWEN_05B_HIDDEN, QWEN_05B_INTER),
        ("Qwen2.5-0.5B MLP down", QWEN_05B_INTER, QWEN_05B_HIDDEN),
        ("Qwen2.5-1.5B attention Q/K/V", QWEN_15B_HIDDEN, QWEN_15B_HIDDEN),
        ("Qwen2.5-1.5B MLP up", QWEN_15B_HIDDEN, QWEN_15B_INTER),
        ("Qwen2.5-1.5B MLP down", QWEN_15B_INTER, QWEN_15B_HIDDEN),
    ]
    rows = []
    for name, din, dout in specs:
        b_star = crossover_batch_size(din, dout)
        rows.append(
            {
                "operation": name,
                "dimensions": f"{din}×{dout}",
                "B_star": int(round(b_star)),
                "regime_at_B1": "Overhead-bound" if b_star > 1 else "Compute-bound",
            }
        )
    return rows


def crossover_demo() -> dict[str, Any]:
    rows = crossover_table()
    min_b = min(r["B_star"] for r in rows)
    return {
        "min_crossover_batch": min_b,
        "all_overhead_bound_at_B1": all(r["B_star"] > 1 for r in rows),
        "sample": rows[1],
    }
