"""Paper tables and anchors (Maczan arXiv:2604.02344)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.torch_webgpu.config import (
    PAPER_ARXIV,
    PAPER_AUTHORS,
    PAPER_REPO,
    PAPER_TITLE,
    PAPER_URL,
    PAPER_VENUE,
    TOK_S_05B_FUSED,
    TOK_S_15B_FUSED,
)

# Table 2 — end-to-end inference (Sec. 4.1)
TABLE2_E2E: list[dict[str, Any]] = [
    {"model": "Qwen2.5-0.5B", "backend": "CUDA compiled RTX 5090", "dtype": "fp16", "tok_s": 185.5, "cv_pct": 0.9, "ttft_ms": 5.4, "vs_cuda": 1.00},
    {"model": "Qwen2.5-0.5B", "backend": "MPS Apple M2", "dtype": "fp16", "tok_s": 47.8, "cv_pct": 0.9, "ttft_ms": 20.9, "vs_cuda": 0.26},
    {"model": "Qwen2.5-0.5B", "backend": "torch-webgpu fused RTX 5090", "dtype": "fp32", "tok_s": 21.0, "cv_pct": 4.0, "ttft_ms": 41.6, "vs_cuda": 0.11},
    {"model": "Qwen2.5-0.5B", "backend": "CPU AMD Ryzen", "dtype": "fp32", "tok_s": 13.7, "cv_pct": 3.2, "ttft_ms": 72.8, "vs_cuda": 0.07},
    {"model": "Qwen2.5-0.5B", "backend": "ONNX Runtime WebGPU", "dtype": "fp32", "tok_s": 13.1, "cv_pct": 1.1, "ttft_ms": 73.5, "vs_cuda": 0.07},
    {"model": "Qwen2.5-1.5B", "backend": "CUDA eager RTX 5090", "dtype": "fp16", "tok_s": 155.3, "cv_pct": 0.6, "ttft_ms": None, "vs_cuda": 1.00},
    {"model": "Qwen2.5-1.5B", "backend": "torch-webgpu fused RTX 5090", "dtype": "fp32", "tok_s": 17.9, "cv_pct": 3.8, "ttft_ms": 51.3, "vs_cuda": 0.12},
    {"model": "Qwen2.5-1.5B", "backend": "torch-webgpu unfused RTX 5090", "dtype": "fp32", "tok_s": 10.4, "cv_pct": 0.9, "ttft_ms": 87.9, "vs_cuda": 0.07},
]

# Table 3 — cross-platform (Sec. 4.2)
TABLE3_CROSS_PLATFORM: list[dict[str, Any]] = [
    {"platform": "Linux", "accelerator": "RTX 5090 CUDA fp16", "tok_s": 185.5, "vs_webgpu": 8.8},
    {"platform": "macOS", "accelerator": "Apple M2 MPS fp32", "tok_s": 12.9, "vs_webgpu": 0.61},
    {"platform": "Windows", "accelerator": "RTX PRO 2000 CUDA fp32", "tok_s": 30.1, "vs_webgpu": 1.4},
    {"platform": "Linux", "accelerator": "AMD Ryzen CPU fp32", "tok_s": 13.7, "vs_webgpu": 0.65},
    {"platform": "Linux", "accelerator": "torch-webgpu RTX 5090 fp32", "tok_s": 21.0, "vs_webgpu": 1.0},
]

# Table 5 — fusion (imported shape from fusion.py progressive)
TABLE5_FUSION: list[dict[str, Any]] = [
    {"configuration": "No fusion", "dispatches_saved": 0, "tok_s": 13.5, "ttft_ms": 71.4},
    {"configuration": "+ Fused RMSNorm", "dispatches_saved": 240, "tok_s": 19.4, "ttft_ms": 46.6},
    {"configuration": "+ Fused MLP", "dispatches_saved": 288, "tok_s": 20.5, "ttft_ms": 43.3},
    {"configuration": "+ Fused K+V", "dispatches_saved": 312, "tok_s": 20.6, "ttft_ms": 41.6},
]

# Table 6 — per-dispatch cost (Sec. 7.2)
TABLE6_DISPATCH: list[dict[str, Any]] = [
    {"implementation": "Dawn RTX 5090", "single_op_us": 496.8, "sequential_us": 23.8, "backend": "Vulkan"},
    {"implementation": "wgpu RTX 5090", "single_op_us": 35.8, "sequential_us": 35.8, "backend": "Vulkan"},
    {"implementation": "wgpu AMD iGPU", "single_op_us": 24.8, "sequential_us": 24.5, "backend": "Vulkan"},
    {"implementation": "wgpu Apple M2", "single_op_us": 48.3, "sequential_us": 71.1, "backend": "Metal"},
    {"implementation": "Chrome RTX 5090 Linux", "single_op_us": 2071.2, "sequential_us": 32.8, "backend": "Vulkan"},
    {"implementation": "Safari Apple M2", "single_op_us": 248.0, "sequential_us": 31.7, "backend": "Metal"},
    {"implementation": "Firefox Apple M2", "single_op_us": 103490.0, "sequential_us": 1038.7, "backend": "Metal", "rate_limited": True},
]

# Table 7 — RMSNorm fusion cross-impl (Sec. 7.3)
TABLE7_RMSNORM_FUSION: list[dict[str, Any]] = [
    {"implementation": "wgpu RTX 5090", "unfused_ms": 0.101, "fused_ms": 0.072, "speedup": 1.41, "backend": "Vulkan"},
    {"implementation": "wgpu AMD iGPU", "unfused_ms": 0.106, "fused_ms": 0.063, "speedup": 1.67, "backend": "Vulkan"},
    {"implementation": "wgpu Apple M2", "unfused_ms": 2.03, "fused_ms": 2.13, "speedup": 0.95, "backend": "Metal"},
    {"implementation": "Chrome RTX 5090", "unfused_ms": 2.11, "fused_ms": 1.99, "speedup": 1.06, "backend": "Vulkan"},
    {"implementation": "Safari Apple M2", "unfused_ms": 0.20, "fused_ms": 0.22, "speedup": 0.91, "backend": "Metal"},
]

# Table 8 — kernel efficiency (Sec. 7.6)
TABLE8_KERNEL_EFF: list[dict[str, Any]] = [
    {"operation": "MLP up projection", "dimensions": "896×896×4864", "time_ms": 6.40, "tflops": 1.22, "pct_peak": 1.2},
    {"operation": "MLP down projection", "dimensions": "896×4864×896", "time_ms": 3.79, "tflops": 2.06, "pct_peak": 2.0},
    {"operation": "Toy matmul", "dimensions": "256×256×256", "time_ms": 1.10, "tflops": 0.030, "pct_peak": 0.1},
]

# Table 9 — optimization recommendations (Sec. 7.8)
TABLE9_RECOMMENDATIONS: list[dict[str, Any]] = [
    {"optimization": "RMSNorm fusion (6→1)", "vulkan": "1.4×", "metal": "0.95×", "notes": "Helps Vulkan only"},
    {"optimization": "Tiled MLP (7→3 disp)", "vulkan": "1.17×", "metal": "2.0×", "notes": "Significant on both"},
    {"optimization": "Command batching", "vulkan": "minimal", "metal": "minimal", "notes": "Sync per token negates benefit"},
]

PAPER_ANCHORS = {
    "per_dispatch_vulkan_us_range": "24–36",
    "per_dispatch_metal_us_range": "31–71",
    "per_operation_overhead_us": "~95",
    "fusion_dispatch_reduction": "876→564",
    "fusion_throughput_gain_pct": 53,
    "torch_webgpu_05b_tok_s": TOK_S_05B_FUSED,
    "torch_webgpu_15b_tok_s": TOK_S_15B_FUSED,
    "cuda_dtype_matched_mobile_ratio": "1.4× WebGPU",
    "single_op_inflation_factor": "~20×",
}


def table2_webgpu_rows() -> list[dict[str, Any]]:
    return [r for r in TABLE2_E2E if "torch-webgpu" in r["backend"]]


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": {
            "arxiv": PAPER_ARXIV,
            "title": PAPER_TITLE,
            "authors": PAPER_AUTHORS,
            "venue": PAPER_VENUE,
            "url": PAPER_URL,
            "repo": PAPER_REPO,
        },
        "table2_e2e": TABLE2_E2E,
        "table3_cross_platform": TABLE3_CROSS_PLATFORM,
        "table5_fusion": TABLE5_FUSION,
        "table6_dispatch": TABLE6_DISPATCH,
        "table7_rmsnorm_fusion": TABLE7_RMSNORM_FUSION,
        "table8_kernel_efficiency": TABLE8_KERNEL_EFF,
        "table9_recommendations": TABLE9_RECOMMENDATIONS,
        "anchors": PAPER_ANCHORS,
    }
