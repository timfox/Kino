"""Per-operation overhead and TTFT accounting (Sec. 4.4, Table 4)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.torch_webgpu.config import (
    DISPATCHES_SAVED_FUSION,
    PER_DISPATCH_DAWN_US,
    PER_OPERATION_OVERHEAD_US,
    TTFT_FUSED_MS,
    TTFT_UNFUSED_MS,
)


def derive_per_operation_overhead_us(
    ttft_unfused_ms: float,
    ttft_fused_ms: float,
    dispatches_saved: int,
) -> float:
    """(TTFT_unfused − TTFT_fused) / dispatches_saved → ~95 μs (Eq. derived Sec. 4.4)."""
    if dispatches_saved <= 0:
        return 0.0
    saved_ms = ttft_unfused_ms - ttft_fused_ms
    return (saved_ms * 1000.0) / dispatches_saved


def framework_overhead_us(per_operation_us: float, per_dispatch_us: float) -> float:
    """Python/framework component = per-operation − per-dispatch."""
    return max(0.0, per_operation_us - per_dispatch_us)


def ttft_accounting(
    *,
    dispatches: int = 564,
    per_dispatch_us: float = PER_DISPATCH_DAWN_US,
    per_operation_us: float = PER_OPERATION_OVERHEAD_US,
    ttft_fused_ms: float = TTFT_FUSED_MS,
) -> dict[str, Any]:
    """Approximate TTFT decomposition (Table 4)."""
    dispatch_ms = dispatches * per_dispatch_us / 1000.0
    framework_ms = dispatches * framework_overhead_us(per_operation_us, per_dispatch_us) / 1000.0
    components_sum = dispatch_ms + framework_ms
    overlap_ms = max(0.0, components_sum - ttft_fused_ms)
    return {
        "ttft_fused_ms": ttft_fused_ms,
        "ttft_unfused_ms": TTFT_UNFUSED_MS,
        "per_operation_overhead_us": per_operation_us,
        "per_dispatch_us": per_dispatch_us,
        "framework_us": framework_overhead_us(per_operation_us, per_dispatch_us),
        "webgpu_dispatch_ms": round(dispatch_ms, 1),
        "framework_ms": round(framework_ms, 1),
        "gpu_cpu_overlap_ms": round(overlap_ms, 1),
        "dispatches": dispatches,
    }


def overhead_demo() -> dict[str, Any]:
    derived = derive_per_operation_overhead_us(
        TTFT_UNFUSED_MS, TTFT_FUSED_MS, DISPATCHES_SAVED_FUSION
    )
    acct = ttft_accounting()
    return {
        "derived_per_operation_us": round(derived, 1),
        "paper_anchor_us": PER_OPERATION_OVERHEAD_US,
        "fusion_speedup_pct": round(
            100.0 * (TTFT_UNFUSED_MS - TTFT_FUSED_MS) / TTFT_UNFUSED_MS, 0
        ),
        "accounting": acct,
    }
