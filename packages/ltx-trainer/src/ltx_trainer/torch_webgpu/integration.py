"""GOPEX integration: browser ML, vLLM stack, and deployment paths."""

from __future__ import annotations

from typing import Any


def gopex_stub_links() -> dict[str, str]:
    return {
        "gopex_vllm_stack": "Native CUDA/MPS baselines vs browser WebGPU path",
        "gopex_portal": "Portal landing — client-side inference demos",
        "gopex_claw": "Multi-channel gateway — optional browser offload",
        "livek12": "K-12 harness — latency-sensitive vs WebGPU tolerance",
        "webllm": "WebLLM q4f16 browser baseline (46–51 tok/s decode)",
        "onnx_runtime_webgpu": "ORT WebGPUExecutionProvider comparison anchor",
    }


def ltx_plan_stub() -> dict[str, Any]:
    return {
        "phase": "webgpu_dispatch_aware_inference",
        "pipeline": [
            "fx_graph_capture_torch_compile",
            "wgsl_shader_codegen_dawn_privateuse1",
            "sequential_dispatch_profiler_validation",
            "kernel_fusion_rmsnorm_mlp_kv",
            "flip_reject_per_token_sync_budget",
            "dtype_matched_baseline_comparison",
        ],
        "representation": "torch.compile FX → WGSL via torch-webgpu",
        "bottleneck": "per-operation overhead ~95 μs at batch=1",
        "scope": "portability-critical browser inference; not CUDA replacement",
    }


def ltx_prep_notes() -> list[str]:
    return [
        "Use sequential-dispatch measurement — single-op benchmarks inflate ~20×.",
        "Prioritize Vulkan fusion (RMSNorm 6→1, MLP 3→1); Metal fusion often neutral.",
        "Short validated pass: 750–1000 iters recovers most compact-chart quality on GPU path.",
        "Batch=1 decode is dispatch-bound (B*≥7); fusion before kernel micro-opts.",
        "Compare dtype-matched float32 CUDA baselines; fp16 gap confounds headline ratios.",
        "Firefox ~1040 μs/dispatch — impractical for ML until rate-limit behavior changes.",
    ]


def integration_bundle() -> dict[str, Any]:
    return {
        "gopex_stubs": gopex_stub_links(),
        "ltx_plan": ltx_plan_stub(),
        "prep_notes": ltx_prep_notes(),
    }
