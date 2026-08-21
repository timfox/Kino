"""FX graph analysis stub (Appendix B, Table 10)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.torch_webgpu.config import (
    DISPATCHES_FUSED,
    DISPATCHES_UNFUSED,
    FX_COMPUTE_OPS,
    FX_SHAPE_OPS,
    FX_TOTAL_NODES,
)


TABLE10_FX_BREAKDOWN: list[dict[str, Any]] = [
    {"category": "Linear (matmul)", "operations": "Q,K,V,O proj, MLP", "count": 169},
    {"category": "Multiply", "operations": "RMSNorm weights, MLP gate", "count": 220},
    {"category": "Add", "operations": "Residuals, biases", "count": 145},
    {"category": "SDPA", "operations": "Attention per layer", "count": 24},
    {"category": "SiLU", "operations": "MLP activation", "count": 24},
    {"category": "RMSNorm components", "operations": "pow, mean, rsqrt", "count": 147},
    {"category": "Concatenation", "operations": "KV cache, rotary", "count": 97},
    {"category": "Other", "operations": "neg, embedding, index", "count": 50},
]


def fx_graph_summary() -> dict[str, Any]:
    compute_sum = sum(r["count"] for r in TABLE10_FX_BREAKDOWN)
    return {
        "total_nodes": FX_TOTAL_NODES,
        "compute_ops": FX_COMPUTE_OPS,
        "shape_ops_no_dispatch": FX_SHAPE_OPS,
        "compute_breakdown_sum": compute_sum,
        "dispatches_upper_bound": DISPATCHES_UNFUSED,
        "dispatches_fused_measured": DISPATCHES_FUSED,
        "breakdown": TABLE10_FX_BREAKDOWN,
    }


def fx_graph_demo() -> dict[str, Any]:
    s = fx_graph_summary()
    return {
        "compute_ops": s["compute_ops"],
        "dispatches_fused": s["dispatches_fused_measured"],
        "top_category": max(s["breakdown"], key=lambda r: r["count"])["category"],
    }
